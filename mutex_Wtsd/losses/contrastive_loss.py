import torch
from torch import nn


def expand_as_one_hot(input_, C, ignore_label=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    NOTE: make sure that the input_ contains consecutive numbers starting from 0, otherwise the scatter_ function
    won't work.

    SPATIAL = DxHxW in case of 3D or SPATIAL = HxW in case of 2D
    :param input_: 3D or 4D label image (NxSPATIAL)
    :param C: number of channels/labels
    :param ignore_label: ignore index to be kept during the expansion
    :return: 4D or 5D output image (NxCxSPATIAL)
    """
    assert input_.dim() in (3, 4), f"Unsupported input shape {input_.shape}"

    # expand the input_ tensor to Nx1xSPATIAL before scattering
    input_ = input_.unsqueeze(1)
    # create result tensor shape (NxCxSPATIAL)
    output_shape = list(input_.size())
    output_shape[1] = C

    if ignore_label is not None:
        # create ignore_label mask for the result
        mask = input_.expand(output_shape) == ignore_label
        # clone the src tensor and zero out ignore_label in the input_
        input_ = input_.clone()
        input_[input_ == ignore_label] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(output_shape).to(input_.device).scatter_(1, input_, 1)
        # bring back the ignore_label in the result
        result[mask] = ignore_label
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(output_shape).to(input_.device).scatter_(1, input_, 1)


def check_consecutive(labels):
    """ Check that the input labels are consecutive and start at zero.
    """
    diff = labels[1:] - labels[:-1]
    return (labels[0] == 0) and (diff == 1).all()


class ContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'

    This implementation expands all tensors to match the instance dimensions.
    This means that it's fast, but has high memory consumption.
    Also, the implementation does not support masking any instance labels in the loss.
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_cluster_means(self, input_, target, ndim):

        dim_arg = (3, 4) if ndim == 2 else (3, 4, 5)

        embedding_dims = input_.size()[1]

        # expand target: NxCxSPATIAL -> # NxCx1xSPATIAL
        target = target.unsqueeze(2)

        # NOTE we could try to reuse this in '_compute_variance_term',
        # but it has another dimensionality, so we would need to drop one axis
        # get number of voxels in each cluster output: NxCx1(SPATIAL)
        num_voxels_per_instance = torch.sum(target, dim=dim_arg, keepdim=True)

        # expand target: NxCx1xSPATIAL -> # NxCxExSPATIAL
        shape = list(target.size())
        shape[2] = embedding_dims
        target = target.expand(shape)

        # expand input_: NxExSPATIAL -> Nx1xExSPATIAL
        input_ = input_.unsqueeze(1)

        # sum embeddings in each instance (multiply first via broadcasting) output: NxCxEx1(SPATIAL)
        embeddings_per_instance = input_ * target
        num = torch.sum(embeddings_per_instance, dim=dim_arg, keepdim=True)

        # compute mean embeddings per instance NxCxEx1(SPATIAL)
        mean_embeddings = num / num_voxels_per_instance

        # return mean embeddings and additional tensors needed for further computations
        return mean_embeddings, embeddings_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance, target, ndim):
        dim_arg = (2, 3) if ndim == 2 else (2, 3, 4)

        # compute the distance to cluster means, result:(NxCxSPATIAL)
        embedding_norms = torch.norm(embeddings_per_instance - cluster_means, self.norm, dim=2)

        # get per instance distances (apply instance mask)
        embedding_norms = embedding_norms * target

        # zero out distances less than delta_var and sum to get the variance (NxC)
        embedding_variance = torch.clamp(embedding_norms - self.delta_var, min=0) ** 2
        embedding_variance = torch.sum(embedding_variance, dim=dim_arg)

        # get number of voxels per instance (NxC)
        num_voxels_per_instance = torch.sum(target, dim=dim_arg)

        # normalize the variance term
        C = target.size()[1]
        variance_term = torch.sum(embedding_variance / num_voxels_per_instance, dim=1) / C
        return variance_term

    def _compute_distance_term(self, cluster_means, C, ndim):
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.

        # squeeze space dims
        for _ in range(ndim):
            cluster_means = cluster_means.squeeze(-1)
        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        cluster_means = cluster_means.unsqueeze(1)
        shape = list(cluster_means.size())
        shape[1] = C

        # NxCxCxExSPATIAL(1)
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(0, 2, 1, 3)
        # compute pair-wise distances (NxCxC)
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=3)

        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        # 1xCxC
        repulsion_dist = repulsion_dist.unsqueeze(0).to(cluster_means.device)
        # zero out distances grater than 2*delta_dist (NxCxC)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        hinged_dist = torch.sum(hinged_dist, dim=(1, 2))
        # normalized by the number of paris and return
        return hinged_dist / (C * (C - 1))

    def _compute_regularizer_term(self, cluster_means, C, ndim):
        # squeeze space dims
        for _ in range(ndim):
            cluster_means = cluster_means.squeeze(-1)
        norms = torch.norm(cluster_means, p=self.norm, dim=2)
        assert norms.size()[1] == C
        # return the average norm per batch
        return torch.sum(norms, dim=1).div(C)

    def forward(self, input_, target):
        """
        Args:
             input_ (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
                                    expects float32 tensor
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)
                                    expects int64 tensor

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """

        n_batches = input_.shape[0]
        # compute the loss per each instance in the batch separately
        # and sum it up in the per_instance variable
        per_instance_loss = 0.
        for single_input, single_target in zip(input_, target):
            # add singleton batch dimension required for further computation
            single_input = single_input.unsqueeze(0)
            single_target = single_target.unsqueeze(0)

            # get number of instances in the batch instance
            instances = torch.unique(single_target)
            assert check_consecutive(instances)
            C = instances.size()[0]

            # SPATIAL = D X H X W in 3d case, H X W in 2d case
            # expand each label as a one-hot vector: N x SPATIAL -> N x C x SPATIAL
            single_target = expand_as_one_hot(single_target, C)

            # compare spatial dimensions
            assert single_input.dim() in (4, 5)
            assert single_input.dim() == single_target.dim()
            assert single_input.size()[2:] == single_target.size()[2:]
            spatial_dims = single_input.dim() - 2

            # compute mean embeddings and assign embeddings to instances
            cluster_means, embeddings_per_instance = self._compute_cluster_means(single_input,
                                                                                 single_target, spatial_dims)
            variance_term = self._compute_variance_term(cluster_means, embeddings_per_instance,
                                                        single_target, spatial_dims)
            distance_term = self._compute_distance_term(cluster_means, C, spatial_dims)
            regularization_term = self._compute_regularizer_term(cluster_means, C, spatial_dims)
            # compute total loss and sum it up
            loss = self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term
            per_instance_loss += loss

        # reduce across the batch dimension
        return per_instance_loss.div(n_batches)
