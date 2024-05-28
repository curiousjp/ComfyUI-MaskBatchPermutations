import torch

class PermuteMaskBatch:
    
    # no internal state
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Input: mask
        """
        return {
            "required": {
                "masks": ("MASK",)
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "permuteMaskBatch"
    OUTPUT_NODE = False
    CATEGORY = "mask"

    def permuteMaskBatch(self, masks):
        n, h, w = masks.shape
        combinations = 2 ** n
        output = torch.zeros((combinations, h, w), dtype=masks.dtype)
        for i in range(1, combinations):
            # exploits the fact that the index is itself a bit pattern 00, 01, 10, 11, etc.
            included = [j for j in range(n) if (i & (1 << j))]
            if included:
                combined = torch.stack([masks[j] for j in included]).max(dim=0)[0]
                output[i] = combined
        return (output,)

class FlattenAgainstOriginal:
    
    # no internal state
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "candidates": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "flattenAgainstOriginal"
    OUTPUT_NODE = False
    CATEGORY = "image"

    def flattenAgainstOriginal(self, base_image, candidates):
        print('inbound shape', base_image.shape)
        target = base_image.clone()
        for b_idx, b in enumerate(target):
            if b.shape[-1] == 4:
                b_rgb = b[..., :3]
                b_alp = b[..., -1:]
            elif b.shape[-1] == 3:
                b_rgb = b
                b_alp = torch.ones(
                    b.shape[:-1] + (1,), 
                    dtype = b.dtype, 
                    device = b.device
                )
            else:
                raise ValueError('final dimension of base images must be 3 or 4')
        
            for c_idx, c in enumerate(candidates):
                c_rgb = c[..., :3]
                c_alp = c[..., -1:]

                new_a = c_alp + (b_alp * (1 - c_alp))
                mask_area = (new_a > 0).squeeze(-1)

                new_rgb = torch.zeros_like(b_rgb)

                new_rgb[mask_area] = (
                    c_rgb[mask_area] * c_alp[mask_area] + 
                    b_rgb[mask_area] * b_alp[mask_area] * (1 - c_alp[mask_area]) 
                ) / new_a[mask_area]

                b_rgb = new_rgb
                b_alp = new_a
            
            if target.shape[-1] == 4:
                target[b_idx] = torch.cat((b_rgb, b_alp), dim = -1)
            else:
                target[b_idx] = b_rgb

        print('return shape', target.shape)
        return (target,)

class CombinatorialDetailer:
    
    # no internal state
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Input: mask
        """
        return {
            "required": {
                "masks": ("MASK",),
                "base_image": ("IMAGE",),
                "candidates": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "combinatorialDetailer"
    OUTPUT_NODE = False
    CATEGORY = "image"

    def combinatorialDetailer(self, masks, base_image, candidates):
        candidate_count, height, width, _ = candidates.shape
        mask_count = masks.shape[0]
        expanded_masks = [x.unsqueeze(-1) for x in masks]

        # Each mask area can be in one of `n + 1` states (all candidates + base)
        num_combinations = (candidate_count + 1) ** mask_count

        output_images = torch.zeros((num_combinations, height, width, 3), dtype=base_image.dtype)
        output_images[0] = base_image[0]

        # Iterate over all other possible combinations
        for i in range(1, num_combinations):
            combined_image = base_image[0].clone()
            current_combination = i
            for mask_index in range(mask_count):
                selected_candidate = current_combination % (candidate_count + 1)
                # print("out image", i, "mask index", mask_index, "selected candidate", selected_candidate)
                if selected_candidate != 0:
                    combined_image = torch.where(expanded_masks[mask_index] == 1, candidates[selected_candidate - 1], combined_image)
                current_combination //= (candidate_count + 1)
            output_images[i] = combined_image

        return (output_images,)


NODE_CLASS_MAPPINGS= {
    "PermuteMaskBatch": PermuteMaskBatch,
    "CombinatorialDetailer": CombinatorialDetailer,
    "FlattenAgainstOriginal": FlattenAgainstOriginal,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PermuteMaskBatch": "Permute Mask Batch",
    "CombinatorialDetailer": "Combinatorial Detailer",
    "FlattenAgainstOriginal": "Flatten Batch against Original",
}

__version__ = "1.1.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


