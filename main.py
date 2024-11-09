import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from guidance.stable import Stable

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def process(prompt, neg_prompt):
  model = Stable()
  # May need to adjust
  lr = 0.1
  gradient_weight = 1
  guidance_weight = 100
  iterations = 1000

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dtype = torch.float16 if torch.cuda.is_available() else torch.float32

  #Embeddings
  text_emb = model.text_embed(prompt, neg_prompt)

  # Intialize latents
  # [TODO] Move initialization to model wrapper
  latents = nn.Parameter(
    torch.rand(
      1, 4, 64, 64,
      dtype=dtype
    )
  )

  # Optimize image
  # May need to add scheduler
  optimizer = torch.optim.Adam(latents, lr=lr)

  for _ in range(iterations):
    # 
    loss = model.sds_loss()
    (1000 * loss).backward()
    optimizer.step()

  # Save image
  

def main():
  prompt = input("Input generation prompt: ")
  neg_prompt = input("Input negative prompt: ")
  save_loc = input("Specify a save locatiion. if it already exists, contents will be overridden: ")
  if not os.path.exists(save_loc):
    os.makedirs(save_loc)
  print("Now Starting SDS: ")
  process(prompt, neg_prompt)

if __name__ == "__main__":
  main()