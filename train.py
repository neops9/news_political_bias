import argparse
import shutil
import torch
import sys
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import structured_attention
from network.data import load_embeddings, read_data, get_input_dict
from network.batch import StandardBatchIterator

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path + filename)
    if is_best:
        shutil.copyfile(path + filename, path + 'model_best.pth.tar')

        
def from_pretrained(args, embeddings_table, unk_idx, n_sources):
    print("Loading pretrained model", flush=True)
    checkpoint = torch.load(args.pretrained, map_location=args.device)
    model = structured_attention.Network(checkpoint['args'], embeddings_table=embeddings_table, word_padding_idx=unk_idx, n_sources=n_sources)
    model_state = model.state_dict()
    model_state.update(checkpoint['state_dict'])
    model.load_state_dict(model_state)
    model.to(device=args.device)
    return model, checkpoint['epoch']
     
   
# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--data", type=str, required=True, help="Path to training data folder")
cmd.add_argument('--embeddings', type=str, required=True, help="Path to the embedding folder")
cmd.add_argument("--model", type=str, default="", help="Path where to store the model")
cmd.add_argument("--pretrained", type=str, default="", help="Path where the pretrained model is")
cmd.add_argument("--lr", type=float, default=0.01)
cmd.add_argument("--weight-decay", type=float, default=0.)
cmd.add_argument("--loss", type=str, help="Which loss to use, if not specified the default loss will be used")
cmd.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
cmd.add_argument('--storage-device', type=str, default="cuda", help="Device to use for data storage")
cmd.add_argument('--device', type=str, default="cuda", help="Device to use for computation")
cmd.add_argument('--resume', type=str, default="", help="Resume training from a saved model.")
cmd.add_argument('--batch-size', type=int, default=16, help="Maximum number of words per batch")
cmd.add_argument('--decay-rate', type=float, default=0.96, help="Decay rate of the ExponentialLR scheduler")
cmd.add_argument('--max-doc-len', type=int, default=-1, help="Maximum number of segments in a document.")
cmd.add_argument('--max-seg-len', type=int, default=-1, help="Maximum number of words in a segment.")
cmd.add_argument('--structured-eval', action="store_true", help="Use structured evaluation")
cmd.add_argument('--seed', type=int, default=1, help='Random seed')
cmd.add_argument('--verbose', action="store_true", help="Verbose")
structured_attention.Network.add_cmd_options(cmd)
args = cmd.parse_args()

torch.manual_seed(args.seed)

print("Loading vocabulary and embeddings", file=sys.stderr, flush=True)
embeddings_table, word_to_id, id_to_word, unk_idx = load_embeddings(args.embeddings)

print("Loading train and dev data", file=sys.stderr, flush=True)
train_data, source_to_id = read_data(args.data, "train", word_to_id, unk_idx, device=args.storage_device, max_seg_length=args.max_seg_len, max_doc_length=args.max_doc_len, min_media_count=10)
train_size = len(train_data)
n_sources = len(source_to_id.keys())
print("Sources: {}".format(source_to_id))
dev_data, _ = read_data(args.data, "dev", word_to_id, unk_idx, device=args.storage_device, max_seg_length=args.max_seg_len+5, max_doc_length=args.max_doc_len+5, min_media_count=10)
dev_size = len(dev_data)
print("Train size: {}".format(train_size))
print("Dev size: {}".format(dev_size))

train_data = StandardBatchIterator(train_data, args.batch_size, shuffle=True)
dev_data = StandardBatchIterator(dev_data, args.batch_size, shuffle=False)

model = structured_attention.Network(args, embeddings_table=embeddings_table, word_padding_idx=unk_idx, n_sources=n_sources)
model.to(device=args.device)

start = 0

if args.pretrained != "":
    model, start_pretrained = from_pretrained(args, embeddings_table, unk_idx, n_sources)
    start = start_pretrained + 1
    
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

# Default loss
loss_builder = nn.CrossEntropyLoss()
loss_builder.to(args.device)

# DANN loss
if args.dann:
    dann_loss_builder = nn.CrossEntropyLoss()
    dann_loss_builder.to(args.device)

best_epoch = 0
best_score = 0

for epoch in range(start, args.epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.
    media_loss_item = 0.
    train_n_correct = 0
    train_media_n_correct = 0
    total_media = 0

    for i, batch in enumerate(train_data):
        optimizer.zero_grad()
        
        input_dict = get_input_dict(batch, unk_idx, args.device)
        
        output = model(input_dict)
        gold = input_dict["gold_labels"]

        if args.dann:
            gold_media = input_dict["media_labels"]
            media_output = model.media_output
            
            media_output = media_output[gold_media!=source_to_id["other"]]
            gold_media = gold_media[gold_media!=source_to_id["other"]]
            
            if media_output.nelement() > 0:
                media_loss = dann_loss_builder(media_output, gold_media)
                media_loss_item += media_loss.item()
                pred_media = media_output.max(1)[1]

            if args.verbose:
                print("Media Prediction:", pred_media, file=sys.stderr, flush=True)
                print("Media Gold:", gold_media, file=sys.stderr, flush=True)
                print("-", file=sys.stderr, flush=True)

        loss = loss_builder(output, gold)
        
        train_loss += loss.item()
        pred = output.max(1)[1]
        
        if args.dann and media_output.nelement() > 0:
            train_media_n_correct += torch.sum(pred_media == gold_media).item()
            total_media += gold_media.shape[0]
            loss += media_loss
        
        
        train_n_correct += torch.sum(pred == gold).item()
        loss = loss / len(batch)

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        
    if not args.model == "":
        remove_list = []

        if not args.train_embs:
            remove_list = ["feature_extractor.word_embs.embs.weight"]

        state_dict = {k: v for k, v in model.state_dict().items() if not k in remove_list}
        save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_score': best_score,
            }, False, args.model, filename=str(epoch) + '.pth.tar')


    # Evaluation
    model.eval()
    dev_loss = 0.
    dev_n_correct = 0
    dev_denum = 0.
    with torch.no_grad():
        for batch in dev_data:
            dev_denum += len(batch)
            input_dict = get_input_dict(batch, unk_idx, args.device)
            output = model(input_dict)
            gold = input_dict["gold_labels"]
            l = loss_builder(output, gold)
            pred = output.max(1)[1]
            dev_n_correct += torch.sum(pred == gold).item()
            dev_loss += l.item()

    print(
        "Epoch %i:"
        "\tTrain loss: %.4f"
        "\t\tTrain acc: %s"
        "\t\tDev loss: %s"
        "\t\tDev acc: %s"
        "\t\tTiming (sec): %i"
        %
        (
            epoch,
            train_loss / train_size,
            train_n_correct / train_size,
            dev_loss / dev_size,
            dev_n_correct / dev_size,
            time.time() - epoch_start_time
        ),
        file=sys.stderr,
        flush=True
    )
