import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_no_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes))


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


def normalize_vis_img(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

## multihead attention network ##
class MultiheadAtt(nn.Module):

    def __init__(self, d_model=64, dim_hider=256, nhead=8, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.dk = dim_hider//nhead
        self.fcq = nn.Linear(d_model, dim_hider)
        self.fck = nn.Linear(d_model, dim_hider)
        self.fcv = nn.Linear(d_model, dim_hider)
        self.fco = nn.Linear(dim_hider, d_model)

        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_hider)
        self.linear2 = nn.Linear(dim_hider, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        ## multihead attention
        dim, nhd, bsz, qsz = self.dk, self.nhead, q.size()[0], q.size()[1]
        qc = q
        q = self.fcq(q).view(bsz, qsz, nhd, dim).permute(2,0,1,3).contiguous().view(-1,qsz,dim)
        k = self.fck(k).view(bsz, qsz, nhd, dim).permute(2,0,1,3).contiguous().view(-1,qsz,dim)
        v = self.fcv(v).view(bsz, qsz, nhd, dim).permute(2,0,1,3).contiguous().view(-1,qsz,dim)

        scores = torch.matmul(q, k.transpose(-2,-1))/(math.sqrt(self.dk))
        attn = self.dropout(F.softmax(scores, dim = -1))
        out = self.fco(torch.matmul(attn,v).view(nhd,bsz,qsz,dim).permute(1,2,0,3).contiguous().view(bsz,qsz,-1)) 

        ## feadfoward network
        qc = qc + self.dropout1(out)
        qc = self.norm1(qc)
        out = self.linear2(self.dropout2(F.relu(self.linear1(qc))))
        qc = qc + self.dropout3(out)
        qc = self.norm2(qc)

        return qc

## attention based feature fusion network ##
class fusion_layer(nn.Module):

    def __init__(self, d_model=64, dim_hider=256, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_att1 = MultiheadAtt(d_model, dim_hider, nhead, dropout)
        self.cross_att2 = MultiheadAtt(d_model, dim_hider, nhead, dropout)
        self.cross_att3 = MultiheadAtt(d_model, dim_hider, nhead, dropout)

    def forward(self, f1, f2):
        bsz, chl, wh = f1.size()[0], f1.size()[1], f1.size()[2]
        f1 = f1.view(bsz,chl,-1).permute(0,2,1).contiguous()
        f2 = f2.view(bsz,chl,-1).permute(0,2,1).contiguous()      
        f11 = self.cross_att1(f1,f2,f2)
        f21 = self.cross_att2(f2,f1,f1)        
        f22 = self.cross_att3(f21,f11,f11)

        return f22.permute(0,2,1).contiguous().view(bsz,chl,wh,wh)

class JcatNet(nn.Module):
    def __init__(self, segm_input_dim=(128,256), segm_inter_dim=(256,256), segm_dim=(64, 64), mixer_channels=2, topk_pos=3, topk_neg=3):
        super().__init__()
        ## segm_input_dim = (64, 256, 512, 1024), segm_inter_dim = (4, 16, 32, 64, 128)
        self.fusion = fusion_layer(d_model=64, dim_hider=512, nhead=8, dropout=0.1) ## attention based fusion net
   
        self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0) ## 1x1,3x3 conv for correlation net
        self.segment1 = conv_no_relu(segm_dim[0], segm_dim[1])

        self.fusion0 = conv(segm_input_dim[3], segm_inter_dim[3], kernel_size=1, padding=0) ## 1x1,3x3 conv for attention net
        self.fusion1 = conv_no_relu(segm_inter_dim[3], segm_inter_dim[3])

        self.mixer0 = conv(mixer_channels, segm_inter_dim[2])    ## correlation net
        self.mixer1 = conv_no_relu(segm_inter_dim[2], segm_inter_dim[3])
        ## mask refinement net 
        self.s3_0 = conv(segm_inter_dim[3], segm_inter_dim[2])
        self.s3_1 = conv_no_relu(segm_inter_dim[2], segm_inter_dim[2])
        self.f3_0 = conv(segm_inter_dim[3], segm_inter_dim[3])
        self.f3_1 = conv_no_relu(segm_inter_dim[3], segm_inter_dim[3])
        self.f2_0 = conv(segm_input_dim[2], segm_inter_dim[3])
        self.f2_1 = conv_no_relu(segm_inter_dim[3], segm_inter_dim[2])
        self.f1_0 = conv(segm_input_dim[1], segm_inter_dim[2])
        self.f1_1 = conv_no_relu(segm_inter_dim[2], segm_inter_dim[1])
        self.f0_0 = conv(segm_input_dim[0], segm_inter_dim[1])
        self.f0_1 = conv_no_relu(segm_inter_dim[1], segm_inter_dim[0])
        self.post2_0 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post2_1 = conv_no_relu(segm_inter_dim[1], segm_inter_dim[1])
        self.post1_0 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post1_1 = conv_no_relu(segm_inter_dim[0], segm_inter_dim[0])
        self.post0_0 = conv(segm_inter_dim[0], 2)
        self.post0_1 = conv_no_relu(2, 2)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, feat_test, feat_train, mask_train, test_dist=None, feat_ups=None, up_masks=None, segm_update_flag=False):

        f_test = self.segment1(self.segment0(feat_test[3]))
        f_train = self.segment1(self.segment0(feat_train[3]))
        ## attention-based features
        f_fusion = self.fusion(self.fusion1(self.fusion0(feat_train[3])), self.fusion1(self.fusion0(feat_test[3])))

        mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1]))
        mask_neg = 1 - mask_pos

        pred_pos, pred_neg = self.similarity_segmentation(f_test, f_train, mask_pos, mask_neg)

        pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
        pred_sm = F.softmax(pred_, dim=-1)[:,:,:,0]
        ## if there are updated segmentation samples
        if segm_update_flag:
            for feat_up, up_mask in zip(feat_ups, up_masks):
                f_up = self.segment1(self.segment0(feat_up[3]))
                up_pos = F.interpolate(up_mask[0], size=(f_up.shape[-2], f_up.shape[-1]))
                up_neg = 1 - up_pos
                pred_up_pos, pred_up_neg = self.similarity_segmentation(f_test, f_up, up_pos, up_neg)
                pred_pos = torch.max(torch.cat((torch.unsqueeze(pred_up_pos, dim=1),torch.unsqueeze(pred_pos, dim=1)), dim=1), dim=1).values
                pred_neg = torch.max(torch.cat((torch.unsqueeze(pred_up_neg, dim=1),torch.unsqueeze(pred_neg, dim=1)), dim=1), dim=1).values
            pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
            pred_sm = torch.max(torch.cat((torch.unsqueeze(F.softmax(pred_, dim=-1)[:,:,:,0], dim=1),torch.unsqueeze(pred_sm, dim=1)), dim=1), dim=1).values

        if test_dist is not None:
            # distance map is given - resize for mixer
            dist = F.interpolate(test_dist[0], size=(f_train.shape[-2], f_train.shape[-1]))
            # concatenate inputs for mixer
            # softmaxed segmentation, positive segmentation and distance map
            segm_layers = torch.cat((torch.unsqueeze(pred_sm, dim=1),
                                     torch.unsqueeze(pred_pos, dim=1),
                                     dist), dim=1)
        else:
            segm_layers = torch.cat((torch.unsqueeze(pred_sm, dim=1), torch.unsqueeze(pred_pos, dim=1)), dim=1)
        out = self.mixer1(self.mixer0(segm_layers)) ## correlation-based features     
        out = self.s3_1(self.s3_0(F.interpolate(self.f3_1(self.f3_0(f_fusion )) + out, scale_factor=2, mode='bilinear', align_corners=False)))
        out = self.post2_1(self.post2_0( F.relu(F.interpolate(self.f2_1(self.f2_0(feat_test[2] )) + out, scale_factor=2, mode='bilinear', align_corners=False))))
        out = self.post1_1(self.post1_0( F.relu(F.interpolate(self.f1_1(self.f1_0(feat_test[1] )) + out, scale_factor=2, mode='bilinear', align_corners=False))))
        out = self.post0_1(self.post0_0( F.relu(F.interpolate(self.f0_1(self.f0_0(feat_test[0] )) + out, scale_factor=2, mode='bilinear', align_corners=False))))

        return out

    ## correlation operation, cosine similarity
    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg):
        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one
        sim = torch.einsum('ijkl,ijmn->iklmn',
                           F.normalize(f_test, p=2, dim=1),
                           F.normalize(f_train, p=2, dim=1))

        sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2], sim.shape[3] * sim.shape[4])

        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
        # re-weight samples (take out positive ang negative samples)
        sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)
        sim_neg = sim_resh * mask_neg.view(mask_neg.shape[0], 1, 1, -1)

        # take top k positive and negative examples
        # mean over the top positive and negative examples
        pos_map = torch.mean(torch.topk(sim_pos, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(torch.topk(sim_neg, self.topk_neg, dim=-1).values, dim=-1)

        return pos_map, neg_map
