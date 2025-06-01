import telebot
import os
import re
import torch
import torchvision
import pandas as pd
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from components.transform import transform

load_dotenv()
token = os.environ.get("TOKEN")

df = pd.read_csv('./src/flowers.csv', index_col='label')
classes = df['name'].to_list()
bot = telebot.TeleBot(token)

model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('./src/flower_classifier.pth'))
model.eval()

@bot.message_handler(commands=['start'])
def start(message):
    html = f'<b>Welcome to Flower Classifier Bot!</b>\nSend me a photo and I\'ll say which one of <b>{len(classes)}</b> flowers in my base it looks like!'
    bot.send_message(message.chat.id, html, parse_mode='html')

@bot.message_handler(content_types=['photo'])
def photo(message):
    reply = bot.reply_to(message, "Loading...")
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image = Image.open(BytesIO(downloaded_file)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
        predicted_class = df.at[predicted.item(), 'name']
        readable_class = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', predicted_class)
        confidence_percent = conf.item() * 100
    bot.edit_message_text(f'That is <b>{readable_class}</b> <i>({confidence_percent:.2f}%)</i>', reply.chat.id, reply.message_id, parse_mode='html')

@bot.message_handler(func=lambda x: True)
def default(message):
    bot.reply_to(message, 'Please send a photo.')

bot.infinity_polling()
