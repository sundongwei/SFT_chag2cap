import timm
import torch
import torch.nn as nn

class CustomViT(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', pretrained=False):
        super(CustomViT, self).__init__()
        
        # Load the original ViT model
        original_vit = timm.create_model(vit_model_name, pretrained=pretrained)
        
        # Custom patch embedding for input shape (32, 2048, 8, 8)
        # Here, we map the input 2048 channels to the ViT embedding dimension (768)
        self.custom_patch_embed = nn.Conv2d(2048, 768, kernel_size=1, stride=1)
        
        # Modify the original ViT model:
        # Replace the original patch embedding and positional embedding layers
        self.cls_token = original_vit.cls_token
        self.pos_drop = original_vit.pos_drop
        self.blocks = original_vit.blocks
        self.norm = original_vit.norm
        
        # Add a layer to project the output back to the original channels
        self.output_proj = nn.ConvTranspose2d(768, 2048, kernel_size=1, stride=1)

    def forward(self, x):
        # Apply the custom patch embedding
        x = self.custom_patch_embed(x)  # Output shape: (32, 768, 8, 8)
        
        # Flatten the spatial dimensions into the sequence dimension
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # Output shape: (32, 64, 768)

        # Add the class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Output shape: (32, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)  # Output shape: (32, 65, 768)
        x = self.pos_drop(x)

        # Forward pass through the transformer blocks
        x = self.blocks(x)  # Output shape: (32, 65, 768)
        x = self.norm(x)  # Output shape: (32, 65, 768)
        
        # Remove the class token and reshape back to (32, 768, 8, 8)
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)  # Output shape: (32, 768, 8, 8)
        
        # Project back to the original channel dimension
        x = self.output_proj(x)  # Output shape: (32, 2048, 8, 8)
        
        return x

# Example usage
if __name__ == "__main__":
    # Create the custom ViT model
    model = CustomViT(vit_model_name='vit_base_patch16_224', pretrained=False)
    
    # Print the architecture to verify
    print(model)
    
    # Define a dummy input with the shape (32, 2048, 8, 8)
    dummy_input = torch.randn(32, 2048, 8, 8)
    
    # Forward pass through the custom model
    output = model(dummy_input)
    print("Output shape:", output.shape)


