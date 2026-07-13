import os
import re

paths = ["api/clip_service.py", "indexer/clip_service.py"]

new_code = """        templates = [
            "a photo of a {} table.",
            "a table with a {} top.",
            "a {} shaped table.",
            "furniture that is {} shaped.",
            "an overhead view of a {} table."
        ]

        with torch.no_grad():
            zeroshot_weights = []
            for label in SHAPE_LABELS:
                texts = [t.format(label) for t in templates]
                class_tokens = clip.tokenize(texts).to(self.device)
                class_embeddings = self.model.encode_text(class_tokens)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            text_features = torch.stack(zeroshot_weights).to(self.device)"""

# Regex to catch the old single-prompt generation and encoding block
pattern = re.compile(
    r'^[ \t]*prompts\s*=\s*\[f"a table with a \{label\} top" for label in SHAPE_LABELS\].*?text_features\s*/=\s*text_features\.norm\(dim=-1,\s*keepdim=True\)',
    re.MULTILINE | re.DOTALL
)

for path in paths:
    if not os.path.exists(path):
        print(f"Skipping {path} - not found.")
        continue
        
    with open(path, "r") as f:
        content = f.read()

    if pattern.search(content):
        updated_content = pattern.sub(new_code, content)
        with open(path, "w") as f:
            f.write(updated_content)
        print(f"Patched {path}")
    else:
        print(f"Could not find target code block in {path}. It may have already been patched or manually altered.")

print("Done.")
