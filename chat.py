from defs import*
import joblib
import openai
from yandexgptlite import YandexGPTLite
import telebot
from telebot.types import InputMediaPhoto
from telebot import types

"""Подгружаем обученные модели"""
model0 = joblib.load('trained_model0.pkl')
model1 = joblib.load('trained_model1.pkl')


"""Подключение yandex gpt"""
account = YandexGPTLite('b1gb95jqeu8n7eddrcj2', 'y0_AgAAAABBg02mAATuwQAAAAED9dsAAAAqc7N7vFVLZYDW0rtPAbFPHwqmUg' )

def GPT (mes,answer1):
    mess="Продолжи представленый диалог своей фразой комментирующей мой ответ.. Диалог был такой:\n Ты спрашиваешь: "+answer1+". \nЯ отвечаю: "+mes+"\nТы отвечаешь: "
    text = account.create_completion(mess, '0.8', system_prompt = 'следует написать только одну фразу для диалога без сторонних фраз')
    print(mess) #Sounds good!
    return text



"""Подключение tg bot"""
bot = telebot.TeleBot('7161422030:AAGOXGc90p98V7qdwrBaD88YUVXaViqvaEQ')

#инициализация
FlagChat =False
previouse_mess="Теперь мы можем спокойно пообщаться)\n Задавай тему!"
answer = ""
waiting_for_answer = False

@bot.message_handler(commands=['start']) #Команда старт дает выбрать режим бота
def startBot(message):
  first_mess = f"<b>{message.from_user.first_name} {message.from_user.last_name}</b>, привет!\n"+ "Я умный чат бот, мне нравится общаться с людьми и распознавать эмоции!"
  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
  btn1 = types.KeyboardButton('/detect') #"""text='\U0001F4F7', callback_data="""
  btn2 = types.KeyboardButton("/chat")
  markup.add(btn1, btn2)
  bot.send_message(message.chat.id, first_mess, parse_mode='html', reply_markup=markup)


"""@bot.callback_query_handler(func=lambda call:True)
def response(function_call):
  if function_call.message:
     if function_call.data == 'send_photo':
        mess = "Отправь мне фото"
        msg=bot.send_message(function_call.message.chat.id, mess)
        bot.register_next_step_handler(msg, handle_photo)"""


@bot.message_handler(commands=['chat'])#Команда чат запускает режим чата
def ChatBot(message):
  #first_mess = f"<b>{message.from_user.first_name} {message.from_user.last_name}</b>, привет!\nХочешь пообщаться?"
  #markup = types.InlineKeyboardMarkup()
  mess = "Теперь мы можем спокойно пообщаться)\n Задавай тему!"
  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
  btn1 = types.KeyboardButton('/end') #"""text='\U0001F4F7', callback_data="""
  markup.add(btn1)
  msg=bot.send_message(message.chat.id, mess, reply_markup=markup)
  bot.register_next_step_handler(msg,handle_text)
  global FlagChat
  global previouse_mess
  FlagChat=True
  previouse_mess = "Теперь мы можем спокойно пообщаться)\n Задавай тему!"


@bot.message_handler(func=lambda message: FlagChat, content_types=['text']) #Генерация и вывод ответа пользователю
def handle_text(message):
    if "/end" in message.text:
        global FlagChat
        FlagChat =False
        startBot(message)
        return

    global previouse_mess
    #mess = str(account.create_completion(str(message.text), '0.8', system_prompt = 'следует написать только одну фразу для диалога без сторонних фраз'))
    mess = GPT(str(message.text),str(previouse_mess))
    bot.send_message(chat_id=message.chat.id, text=mess)
    previouse_mess = mess




@bot.message_handler(commands=['detect']) #Команда запускающая режим определения эмоций
def DetectBot(message):

    mess = "Отправь мне фото"
    markup = types.ReplyKeyboardMarkup()
    item_btn = types.KeyboardButton('/end')
    markup.row(item_btn)
    msg = bot.send_message(chat_id=message.chat.id, text=mess, reply_markup=markup)
    bot.register_next_step_handler(msg, handle_photo)
    """with open('path_to_photo', 'rb') as photo:
       bot.send_media_group(function_call.message.chat.id, [InputMediaPhoto(photo)])
    # bot.answer_callback_query(function_call.id)"""

@bot.message_handler(content_types=['photo','text']) #Обработка фото и вывод результата
def handle_photo(message):

    try:
        # Скачивание фото
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open("image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
            # Получение предсказания
        prediction = predict(["image.jpg"], model1)
        answer1, answer2 = get_message(prediction[0])

        markup = types.ReplyKeyboardMarkup()
        item_btn = types.KeyboardButton('/end')
        markup.row(item_btn)
        msg=bot.send_message(chat_id=message.chat.id, text=answer1, reply_markup=markup)

        global answer
        answer = answer1
        global FlagChat
        FlagChat = False
        global waiting_for_answer
        waiting_for_answer = True

        bot.register_next_step_handler(msg,handle_answer)

    except:
        if "/end" in message.text:
            # global waiting_for_answer
            waiting_for_answer = False
            startBot(message)
            return
        DetectBot(message)



@bot.message_handler(func=lambda message: waiting_for_answer, content_types=['text']) # Ожидание ответа от пользователя
def handle_answer(message, answer=answer):

    global waiting_for_answer

    if "/end" in message.text:
        #global waiting_for_answer
        waiting_for_answer = False
        startBot(message)
        return

    try:
        bot.send_message(message.chat.id, str(GPT(message.text, answer)))
        DetectBot(message)
        # Сброс флага, чтобы вернуться к обработке любых сообщений
        waiting_for_answer = False
    except:
        handle_photo(message)
        return

# Запуск бота
bot.polling()


""""bot в терминале"
def ChekOut (mes):
    if mes=="пока":
        return True
    return False
print("-------------------Чат бот---------------------")
Exit=False
while (not(Exit)):
    print("------------------Новый диалог-----------------")
    print("Bot:  Привет, как ты? Покажись..")
    MYface = input()
    Exit = ChekOut(MYface)
    if Exit:
        break
    print("Я:  Можешь посмотреть моё фото -",MYface)
    #answer2=""
    if check_jpg_file(MYface):
        print("Bot:  Путь указывает на файл формата JPG.")
        emodzi = predict([MYface],model1)
        answer1, answer2 = get_message(emodzi[0])
        print ("Bot: ",emodzi,answer1)
        mes = input()
        Exit = ChekOut(MYface)
        print("Я:  ", mes)  
        # print("Bot:  ",answer2)
        print("Bot:  ", GPT(mes, answer1))
    else:
        print("Bot:  Это не похоже на путь к фото, вероятно ты ошибся..")
"""