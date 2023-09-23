import math
import torch.nn as nn
import torch
import numpy as np
import itertools
import torch.nn.functional as F
from torch.autograd import Function
import network.special_tokens

class SequenceDropout(nn.Module):
    def __init__(self, p, broadcast_batch=False, broadcast_segment=False, broadcast_word=False):
        super().__init__()
        self.p = p
        self.broadcast_batch = broadcast_batch # samples a dropout mask that is constant in the batch dimension
        self.broadcast_segment = broadcast_segment # samples a dropout mask that is constant in the segment dimension
        self.broadcast_word = broadcast_word # samples a dropout mask that is constant in the word dimension

    # (n batch, n segments, n words, n features)
    def forward(self, x):
        if self.training and self.p > 0.:
            # assume x is of shape (batch, segment, word, feature)
            n_batch = 1 if self.broadcast_batch else x.shape[0]
            n_segments = 1 if self.broadcast_segment else x.shape[1]
            n_words = 1 if self.broadcast_word else x.shape[2]
            n_feats = x.shape[3]
            mask = torch.empty(
                (n_batch, n_segments, n_words, n_feats),
                dtype=torch.float,
                device=x.device,
                requires_grad=False
            )
            mask.bernoulli_(1 - self.p)
            x = x * mask
        return x


class BiLSTM(nn.Module):
    def __init__(self, args, input_size, hidden_size, num_layers=1, dropout=0.5, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.device = args.device

        # if bidirectional, forward and backward outputs are concatenated
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)

    def forward(self, input):
        sent_output, (hidden, cell) = self.bilstm(input)
        return sent_output, hidden

    def forward_packed(self, input, seq_lengths):
        # Clamp everything to minimum length of 1, but keep the original variable to mask the output later
        seq_lengths_clamped = seq_lengths.clamp(min=1, max=input.shape[1])
        sent_packed = nn.utils.rnn.pack_padded_sequence(input, seq_lengths_clamped.cpu(), batch_first=True, enforce_sorted=False)
        sent_output_packed, (hidden, cell) = self.bilstm(sent_packed)
        sent_output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(sent_output_packed, batch_first=True)

        # mask everything that had seq_length as 0 in input as 0 (mask padded segments)
        mask = (seq_lengths == 0).view(-1, 1, 1)
        sent_output_padded.masked_fill_(mask, 0)

        return sent_output_padded, hidden


class SpecialEmbeddingsNetwork(nn.Module):
    def __init__(self, size):
        super(SpecialEmbeddingsNetwork, self).__init__()
        self.size = size

        self.dict = network.special_tokens.Dict()
        self.embs = nn.Embedding(len(self.dict)+1, self.size, padding_idx=len(self.dict)) 

        print("Special word embeddings size: %i" % size)
        print(flush=True)

    def forward(self, inputs):
        values = self.embs(inputs)
        return values


class Glove(torch.nn.Module):
    def __init__(self, args, embedding_table, word_padding_idx):
        super().__init__()
        n_embs = len(embedding_table)
        self.embs = nn.Embedding(n_embs+1, args.word_embs_dim, padding_idx=word_padding_idx)

        # Freeze words embs
        self.embs.weight.requires_grad=args.train_embs
        
        self.special_embs = SpecialEmbeddingsNetwork(args.word_embs_dim)
        self.n_unk = len(network.special_tokens.Dict())
        self.add_context = args.add_context
        self.output_dim = args.word_embs_dim

        with torch.no_grad():
            for i, v in enumerate(embedding_table):
                self.embs.weight[i, :] = torch.tensor(embedding_table[v])

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--train-embs', action="store_true", help="Train embeddings")

    def forward(self, inputs):
        # REMETTRE SPECIAL EMBEDDINGS
        repr_list = [self.embs(inputs["token_idxs"])]# + self.special_embs(inputs["special_words_idxs"])]

        if len(repr_list) == 1:
            ret = repr_list[0]
        else:
            ret = torch.cat(repr_list, 2)

        return ret


class FeatureExtractionModule(nn.Module):
    def __init__(self, args, embeddings_table, word_padding_idx):
        super(FeatureExtractionModule, self).__init__()
        self.output_dim = 0

        if args.word_embs:
            self.word_embs = Glove(args, embeddings_table, word_padding_idx)
            self.output_dim += self.word_embs.output_dim
        else:
            self.word_embs = None

        if self.output_dim == 0:
            raise RuntimeError("No input features set!")

    @staticmethod
    def add_cmd_options(cmd):
        Glove.add_cmd_options(cmd)

        cmd.add_argument('--word-embs', action="store_true", help="Use word embeddings")
        cmd.add_argument('--word-embs-dim', type=int, default=300, help="Dimension of the word embs (Glove = 300)")

    def forward(self, inputs):
        repr_list = []

        if self.word_embs is not None:
            ret = self.word_embs(inputs)
            repr_list.append(ret)

        if len(repr_list) == 1:
            token_repr = repr_list[0]
        else:
            token_repr = torch.cat(repr_list, 2)

        return token_repr


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class StructuredAttentionModule(nn.Module):
    def __init__(self, args):
        super(StructuredAttentionModule, self).__init__()
        # x2 multipliers are used to handle the bidirectional LSTM which has 2 outputs (forward/backward outputs)
        self.device = args.device
        self.sem_dim = args.sem_dim
        self.struct_dim = args.struct_dim
        self.tree_percolation = args.tree_percolation

        # Unnormalized attention scores F_ij
        self.tp_linear = nn.Linear(2*self.struct_dim, 2*self.struct_dim, bias=True) # representation of parent nodes
        self.tc_linear = nn.Linear(2*self.struct_dim, 2*self.struct_dim, bias=True) # representation of child nodes 
        self.t_activation = nn.Tanh()
        self.fij_bilinear = nn.Linear(2*self.struct_dim, 2*self.struct_dim, bias=False)
        self.f_i_root_linear = nn.Linear(2*self.struct_dim, 1, bias=False)
        
        # Special embedding for the root node
        self.embeddings_root_s = nn.Parameter(torch.Tensor(1, 1, 2*self.sem_dim))

        # Update semantic vector
        self.r_i_linear = nn.Linear(2 * 2 * self.sem_dim, 2 * self.sem_dim, bias=True)
        self.r_i_activation = nn.LeakyReLU(negative_slope=0.01)

        # Tree percolation
        if self.tree_percolation:
            self.percolation_linear = nn.Linear(2 * 2 * self.sem_dim, 2 * self.sem_dim, bias=True)

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.tp_linear.weight)
            torch.nn.init.xavier_uniform_(self.tc_linear.weight)
            torch.nn.init.xavier_uniform_(self.fij_bilinear.weight)
            torch.nn.init.xavier_uniform_(self.f_i_root_linear.weight)
            torch.nn.init.xavier_uniform_(self.embeddings_root_s)
            torch.nn.init.xavier_uniform_(self.r_i_linear.weight)
            torch.nn.init.xavier_uniform_(self.percolation_linear.weight)
            self.r_i_linear.bias.zero_()
            self.tp_linear.bias.zero_()
            self.tc_linear.bias.zero_()
            self.percolation_linear.bias.zero_()

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--sem-dim', type=int, default=50, help="Dimension of the semantic vector")
        cmd.add_argument('--struct-dim', type=int, default=50, help="Dimension of the structure vector")
        cmd.add_argument('--tree-percolation', type=int, default=-1, help="Additional level of percolation over the marginals to incorporate the children’s children of the tree.")

    def get_matrix_tree(self, f_ij, f_i_root, seq_lengths, dim_batch, dim_token, dependencies=None, masked_dependencies=None):
        # Compute A_ij and A_i_root matrices
        A_ij = torch.exp(f_ij)
        torch.diagonal(A_ij, dim1=1, dim2=2).zero_() # set diagonal to 0 (i = j)
        
        A_i_root = torch.exp(f_i_root)
        
        # Masking padded values
        mask = torch.zeros(A_ij.shape[0], A_ij.shape[1] + 1, dtype=A_ij.dtype, device=A_ij.device)
        mask[(torch.arange(A_ij.shape[0]), seq_lengths)] = 1
        mask = mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column
        A_ij = A_ij * (1. - mask[..., None]) # mask first dimension
        A_ij = (A_ij.transpose(2, 1) * (1. - mask[..., None])).transpose(2, 1) # mask second dimension
        A_i_root = A_i_root * (1. - mask) # mask root

        # Compute de Laplacian matrix L_ij of the graph
        # if i = j: sum_{i'=1}{n} A_i'j
        i_eq_j = torch.diag_embed(torch.sum(A_ij, dim=1))
        # otherwise : -A_ij
        L_ij = i_eq_j - A_ij

        # Compute L_ij_bar, a variant of L_ij that take the
        # root node into consideration: f_i_root if i = 0; L_ij otherwise
        L_ij_bar = L_ij.detach().clone()
        L_ij_bar[:,0,:] = A_i_root.detach().clone()

        # Set the padded diagonals to 1 for the matrix to be matrix invertible
        diag_mask = torch.diag_embed(mask)
        L_ij_bar += diag_mask

        # Compute the inverse matrix of L_ij_bar that will be use to compute marginal probabilities
        L_ij_bar_inv = torch.inverse(L_ij_bar)

        # Compute the marginal probability of the word i to be the root
        p_root = A_i_root * L_ij_bar_inv[:, :, 0]
        
        # Compute the marginal probability of the dependency edge 
        # between the i-th and j-th words.
        # This can be interpreted as attention scores which are
        # constrained to converge to a non-projective dependency tree

        # See equation (15) in Liu&Lapata
        # (1 - delta_1_j) * A_ij * [L_ij_bar_inv]_j_j
        L_ij_bar_inv_diag = torch.diagonal(L_ij_bar_inv, dim1=-2, dim2=-1).unsqueeze(2)
        part1 = (A_ij.transpose(1,2) * L_ij_bar_inv_diag).transpose(1,2)

        # (1 - delta_i_1) * A_ij * [L_ij_bar_inv]_j_i
        part2 = A_ij * L_ij_bar_inv.transpose(1,2)

        # Masking values following the Kronecker delta
        tmp1 = torch.zeros(dim_batch, dim_token, 1)
        tmp2 = torch.zeros(dim_batch, 1, dim_token)
        mask1 = torch.ones(dim_batch, dim_token, dim_token-1)
        mask2 = torch.ones(dim_batch, dim_token-1, dim_token)
        # (1 - delta_1_j)
        mask1 = torch.cat([tmp1, mask1], 2).to(self.device)
        # (1 - delta_i_1)
        mask2 = torch.cat([tmp2, mask2], 1).to(self.device)

        p_ij = mask1 * part1 - mask2 * part2
        
        # add vector at beginning for root scores
        p_ij_root = torch.cat([p_root.unsqueeze(1), p_ij], dim=1) # (batch * segments) * tokens+1 * tokens        
        
        return p_ij, p_ij_root

    def forward(self, input, seq_lengths, dependencies=None, masked_dependencies=None): 
        # if tokens_attention: (batch * segments) * tokens * hidden
        dim_batch, dim_token, dim_hidden = input.shape
        input = input.view(dim_batch, dim_token, 2, dim_hidden//2)

        # get semantic and structure vector from forward/backward biLSTM outputs
        semantic_vector = torch.cat((input[:, :, 0, :self.sem_dim], input[:, :, 1, :self.sem_dim]), 2)
        structure_vector = torch.cat((input[:, :, 0, self.sem_dim:], input[:, :, 1, self.sem_dim:]), 2)

        # Representations of parent and child nodes
        tp = self.t_activation(self.tp_linear(structure_vector)) # (batch * segments), tokens, struct_dim
        tc = self.t_activation(self.tc_linear(structure_vector)) # (batch * segments), tokens, struct_dim

        # Compute unnormalized attention scores F_ij (between token i and j) using bilinear transformation
        tmp = torch.tensordot(tp, self.fij_bilinear.weight, [[-1],[0]]) # tp.Wa
        f_ij = torch.matmul(tmp, torch.transpose(tc, 2, 1)) # tp.Wa.tc^T

        # Compute root score f_i^r for each element
        f_i_root = self.f_i_root_linear(structure_vector).squeeze()

        # Given unnormalized attention scores F_ij and root scores F_i^r,
        # compute the marginal probability of dependency edges + root node
        # using a variant of Kirchhoff’s Matrix-Tree Theorem
        struct_scores, struct_scores_with_root = self.get_matrix_tree(f_ij.detach().clone(), f_i_root.detach().clone(), seq_lengths, dim_batch, dim_token, dependencies, masked_dependencies)

        # Update semantic vectors with structured attention (see eq. (16) Liu&Lapata)
        struct_scores_p = torch.transpose(struct_scores_with_root.detach().clone(), 1, 2) # soft parent

        # TODO: how the sum is done here ?
        # Compute context vector gathered from possible parents
        # p_i = sum_{k=0}{n} (a_ki*e_k) +  a_i^r * e_root 
        # (e_root is a special embedding for the root node)
        sem_root = torch.cat([torch.tile(self.embeddings_root_s, [dim_batch, 1, 1]), semantic_vector], 1)
        parents_ = torch.matmul(struct_scores_p, sem_root)

        # Compute context vector gathered from possible children
        # c_i = sum_{k=0}{n} (a_ik*e_i) 
        children_ = torch.matmul(struct_scores, semantic_vector)

        # Compute the updated semantic vector r_i, with rich structural information (eq. 18 Liu&Lapata)
        output_ = self.r_i_activation(self.r_i_linear(torch.cat([parents_, children_], dim=2)))
        
        # Percolation is only supported for childrens
        if self.tree_percolation > 0:
            for i in range(self.tree_percolation):
                children_perc = torch.matmul(struct_scores, output_)
                output_ = self.r_i_activation(self.percolation_linear(torch.cat([output_, children_perc], dim=2)))

        return output_, struct_scores_with_root
        

class Network(nn.Module):
    def __init__(self, args, embeddings_table, word_padding_idx, n_sources=-1):
        super(Network, self).__init__()

        torch.autograd.set_detect_anomaly(True)
        
        self.device = args.device
        self.sem_dim = args.sem_dim
        self.struct_dim = args.struct_dim
        self.skip_structured_attention = args.skip_structured_attention
        self.dann = args.dann

        # Feature extraction
        self.feature_extractor = FeatureExtractionModule(args, embeddings_table, word_padding_idx)
        self.input_dropout = SequenceDropout(p=args.input_dropout, broadcast_batch=False, broadcast_segment=False, broadcast_word=False)

        # LSTMs
        self.lstm_hidden_dim = args.sem_dim + args.struct_dim
        self.words_lstm = BiLSTM(args, self.feature_extractor.output_dim, self.lstm_hidden_dim, dropout=args.input_dropout)
        self.docs_lstm = BiLSTM(args, 2 * self.sem_dim, self.lstm_hidden_dim, dropout=args.input_dropout)

        # Structured Attention
        self.segments_structured_attention = StructuredAttentionModule(args)
        self.documents_structured_attention = StructuredAttentionModule(args)

        self.word_embs_dim = args.word_embs_dim

        # Skip structured attention 
        self.skip_struct_linear = nn.Linear((2 * self.lstm_hidden_dim), 2 * self.sem_dim, bias=True)
        self.skip_struct_activation = nn.LeakyReLU(negative_slope=0.01)
        #self.skip_struct_activation = nn.Tanh()

        # Final output
        if self.skip_structured_attention:    
            dim_tmp = 2*self.lstm_hidden_dim
        else:
            dim_tmp = 2 * self.sem_dim

        self.final_output = nn.Sequential(
            nn.Linear(dim_tmp, args.proj_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.Tanh(),
            nn.Dropout(p=args.proj_dropout),
            nn.Linear(args.proj_dim, args.dim_output, bias=True)
        )

        # Domain Adaptation
        if self.dann:
            self.dann_output = nn.Sequential(
                #nn.Linear(dim_tmp, args.proj_dim, bias=True),
                nn.Linear(2*self.lstm_hidden_dim, args.proj_dim, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                #nn.Tanh(),
                nn.Dropout(p=args.proj_dropout),
                nn.Linear(args.proj_dim, n_sources, bias=True)
            )
            self.dann_alpha = args.dann_alpha

        self.documents_attention_scores_with_root = None
        self.segments_attention_scores_with_root = None

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.final_output[0].weight)
            torch.nn.init.xavier_uniform_(self.final_output[3].weight)
            self.final_output[0].bias.zero_()
            self.final_output[3].bias.zero_()

            if self.dann:
                torch.nn.init.xavier_uniform_(self.dann_output[0].weight)
                torch.nn.init.xavier_uniform_(self.dann_output[3].weight)
                self.dann_output[0].bias.zero_()
                self.dann_output[3].bias.zero_()

            torch.nn.init.xavier_uniform_(self.skip_struct_linear.weight)
            self.skip_struct_linear.bias.zero_()

    def forward(self, input, dependencies=None, dann_alpha=0.7):
        feature = self.feature_extractor(input)
        feature = self.input_dropout(feature)

        # store useful variables
        batch_l = input['batch_l']
        max_doc_l = input['max_doc_l']
        max_segment_l = input['max_segment_l']
        segments_l = input['segments_l']
        doc_l = input['doc_l']

        # Masked dependencies for structure explanation
        masked_dependencies = input['mask_heads_matrix']

        # segments mask for padded segments
        mask_segments = input['mask_segments'][:, :max_doc_l]

        # Reshape as: (batch_l * max_doc_l, max_segment_l, word_embs_dim)
        feature = torch.reshape(feature, (batch_l * max_doc_l, max_segment_l, self.word_embs_dim))

        encoded_segments, _ = self.words_lstm.forward_packed(feature, torch.flatten(segments_l))
        
        seg_l = torch.flatten(segments_l)
        d_l = torch.flatten(doc_l)

        if not self.skip_structured_attention:
            encoded_segments = encoded_segments.sum(dim=1)
            seg_l_bool = seg_l.clone()
            seg_l_bool[seg_l_bool==0] = 1
            encoded_segments = torch.div(encoded_segments, seg_l_bool.unsqueeze(1))
            
            encoded_segments = torch.reshape(encoded_segments, [batch_l, max_doc_l, 2 * self.lstm_hidden_dim])
            encoded_documents, documents_attention_scores_with_root = self.documents_structured_attention(encoded_segments, torch.flatten(doc_l), dependencies, masked_dependencies)
            self.documents_attention_scores_with_root = documents_attention_scores_with_root

            encoded_documents = encoded_documents.sum(dim=1)
            encoded_documents = torch.div(encoded_documents, doc_l.unsqueeze(1))            
        else:
            encoded_documents = torch.reshape(encoded_segments, [batch_l, max_doc_l * max_segment_l, 2 * self.lstm_hidden_dim])
            encoded_documents = encoded_documents.sum(dim=1)

        if self.dann:
            encoded_documents_dann = encoded_segments.sum(dim=1)
            encoded_documents_dann = torch.reshape(encoded_documents_dann, [batch_l, max_doc_l, 2 * self.lstm_hidden_dim])
            encoded_documents_dann = encoded_documents_dann.sum(dim=1)
            reversed_encoded = ReverseLayerF.apply(encoded_documents_dann, dann_alpha)

        output = self.final_output(encoded_documents)

        if self.dann:
            self.media_output = self.dann_output(reversed_encoded)

        return output

    @staticmethod
    def add_cmd_options(cmd):
        FeatureExtractionModule.add_cmd_options(cmd)
        StructuredAttentionModule.add_cmd_options(cmd)
        
        cmd.add_argument('--dim-output', type=int, default=3, help="Number of labels")
        cmd.add_argument('--proj-dim', type=int, default=128, help="Dimension of the output projection")
        cmd.add_argument('--activation', type=str, default="tanh", help="activation to use in weightning modules: tanh, relu, leaky_relu")
        cmd.add_argument('--input-dropout', type=float, default=0.0, help="Dropout for the input")
        cmd.add_argument('--proj-dropout', type=float, default=0.0, help="Dropout for the proj")
        cmd.add_argument('--add-context', action="store_true", help="Add word's left and right context")
        cmd.add_argument('--dann', action="store_true", help="Use domain adaptation (media adaptation)")
        cmd.add_argument('--dann-alpha', type=float, default=0.7, help="Alpha parameter of the dann module")
        cmd.add_argument('--skip-structured-attention', action="store_true", help="Skip structured attention")
