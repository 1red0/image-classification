import asyncio

model_lock = asyncio.Lock()
model = None

img_height = 256
img_width = 256