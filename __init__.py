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

NODE_CLASS_MAPPINGS= {
    "PermuteMaskBatch": PermuteMaskBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PermuteMaskBatch": 'Permute Mask Batch'
}

__version__ = '1.0.0'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


