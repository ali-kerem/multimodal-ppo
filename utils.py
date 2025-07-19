import torch

def get_pixel_values(queries, micro_batch_inds):
    pixel_values = []
    for i in range(len(micro_batch_inds)):
        pixel_values_start = torch.sum(torch.prod(queries["image_grid_thw"][:micro_batch_inds[i]],dim=1))
        pixel_values_end = torch.sum(torch.prod(queries["image_grid_thw"][:micro_batch_inds[i]+1],dim=1))
        pixel_values.append(queries["pixel_values"][pixel_values_start:pixel_values_end])
    pixel_values = torch.cat(pixel_values, dim=0)
    return pixel_values