import streamlit as st
import torch
import torchvision

st.title("Test PyTorch en Streamlit Cloud")

st.write("torch version:", torch.__version__)
st.write("torchvision version:", torchvision.__version__)
st.write("cuda available?", torch.cuda.is_available())
x = torch.randn(3,3)
st.write("tensor sample:", x)
