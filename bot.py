import os
import io
import asyncio
import logging
import re
from datetime import datetime
import aiohttp
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BotCommandScopeDefault, BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
# Для вызова агентов и LLM
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Забираем токен для инициализации бота
load_dotenv()
logging.basicConfig(level=logging.INFO)
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())
users: dict[int, dict] = {} # пока без БД, сорри(
# Мы продвинутые и суем агентов везде, поэтому пусть этот бот не станет исключением
# Инициализируем LLM один раз
llm = ChatOpenAI(
    openai_api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model_name=os.getenv("LLM_MODEL_NAME")
)

# Напишем Pydantic-модели для структурированного выхода LLM
class FoodInfo(BaseModel):
    name: str = Field(description="Название продукта")
    calories_per_100g: float = Field(description="Калорийность на 100 грамм")

class WorkoutEstimate(BaseModel):
    burned_calories: int = Field(description="Сожженные калории")
    bonus_water_ml: int = Field(description="Дополнительная норма воды (мл)")

class ProfileStates(StatesGroup):
    weight = State()
    height = State()
    age = State()
    high_intensity = State()
    low_intensity = State()
    city = State()

class LogWaterStates(StatesGroup):
    waiting_for_amount = State()

class FoodLogStates(StatesGroup):
    waiting_for_product = State()
    waiting_for_amount = State()
    waiting_for_unit = State()
    waiting_for_piece_grams = State()
    confirm_more = State()

class WorkoutStates(StatesGroup):
    choose_intensity = State()
    waiting_for_input = State()

# Перейдем к самому соку (напишем команды для нашего бота)
async def set_commands():
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="set_profile", description="Настроить профиль (сделай это сразу!)"),
        BotCommand(command="info", description="Как пользоваться ботом (описание команд)"),
        BotCommand(command="log_water", description="Записать потребление воды"),
        BotCommand(command="log_food", description="Записать потребление еды"),
        BotCommand(command="log_workout", description="Записать информацию о тренировке"),
        BotCommand(command="check_progress", description="Узнать потребление калорий и воды"),
        BotCommand(command="stats", description="График прогресса по воде и калориям"),
        BotCommand(command="recommendations", description="Получить рекомендации по питанию / тренировкам"),
        BotCommand(command="reset_day", description="Сброс дня (тест: обнулить каллории, воду, активность)"),
        BotCommand(command="who_am_i", description="Кто я, что я могу, и зачем я нужен?"),
        BotCommand(command="disclaimer", description="Дисклеймер (не полагайся полностью на бота!)"),
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())

# Функции, которые будут нам очень помогать
# Обрабатываем кейсы когда пользователь ввел явно не то, что нужно (условно возраст не может быть отрицательным)
def validate_range(value, min_v, max_v):
    return min_v <= value <= max_v

# Чтобы получить погоду
async def get_weather(city: str) -> float:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return 20

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                return data["main"]["temp"]
    except Exception:
        logging.exception("Weather error")
        return 20

# КОСТЫЛЬ: по умолчанию дневные нормы пользователя сбрасываются каждый день, но при тестировании решения 
# должна быть опция мгновенного сброса
def check_daily_reset(uid: int):
    user = users.get(uid)
    if not user:
        return

    today = datetime.now().date()
    if user["last_reset"] != today:
        user.update({
            "logged_water": 0,
            "logged_calories": 0,
            "burned_calories": 0,
            "daily_water_adjustment": 0,
            "today_high_minutes": 0,
            "today_low_minutes": 0,
            "water_norm": user["base_water_norm"],
            "last_reset": today
        })

async def get_food_info(product: str) -> FoodInfo | None:
    parser = PydanticOutputParser(pydantic_object=FoodInfo)
    prompt = PromptTemplate(
        template=(
            "Определи калорийность продукта.\n"
            "Продукт: {product}\n"
            "{format_instructions}"
        ),
        input_variables=["product"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    try:
        chain = prompt | llm | parser
        return await chain.ainvoke({"product": product})
    except Exception:
        logging.exception("LLM food error")
        return None

async def estimate_piece_weight(product: str) -> int | None:
    prompt = (
        f"Оцени средний вес одной штуки продукта '{product}' в граммах. "
        "Отвечай одним целым числом (только число, без текста)."
    )
    try:
        res = await llm.ainvoke(prompt)
        text = getattr(res, "content", str(res)).strip()
        m = re.search(r"(\d{1,4})", text.replace(",", ""))
        if m:
            val = int(m.group(1))
            if 5 <= val <= 2000:
                return val
        nums = re.findall(r"\d+", text)
        for n in nums:
            val = int(n)
            if 5 <= val <= 2000:
                return val
        return None
    except Exception:
        logging.exception("Ошибка определения веса ААААА")
        return None

def estimate_burned_calories_det(workout: str, minutes: int, weight: float) -> int:
    # напишем парочку костылей
    lw = workout.lower()
    if any(k in lw for k in ["бег"]):
        met = 10
    elif any(k in lw for k in ["плав"]):
        met = 9
    elif any(k in lw for k in ["бокс"]):
        met = 10
    elif any(k in lw for k in ["вел"]):
        met = 8
    elif any(k in lw for k in ["ход"]):
        met = 3
    elif any(k in lw for k in ["йог"]):
        met = 2.5
    else:
        met = 6
    burned = int(met * weight * (minutes / 60.0))
    return burned

def classify_workout_intensity(workout: str) -> str:
    lw = workout.lower()
    high_keywords = ["бег", "плав", "бокс", "интенсив", "инт", "спринт", "вел", "тяжелая"]
    low_keywords = ["ход", "йог", "растяж", "стретч", "легк", "пеш", "прогул"]
    if any(k in lw for k in high_keywords):
        return "high"
    if any(k in lw for k in low_keywords):
        return "low"
    # считаем как low по умолчанию
    return "low"

def unit_choice_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="граммы")], [KeyboardButton(text="штуки")]], resize_keyboard=True
    )

def more_food_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="+ Добавить еще")], [KeyboardButton(text="✅ Завершить")]], resize_keyboard=True
    )

def intensity_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Высокая")], [KeyboardButton(text="Низкая")]], resize_keyboard=True
    )

def main_menu_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="set_profile")],], resize_keyboard=True, one_time_keyboard=False
    )

# жесть у нас с вами клавиатур как будто мы не ботов пишем а магазин с электроникой открываем

# def after_set_profile_keyboard():
#     return ReplyKeyboardMarkup(
#         keyboard=[
#             [KeyboardButton(text="log_water"), KeyboardButton(text="log_food"), KeyboardButton(text="log_workout"), KeyboardButton(text="info"),
#              KeyboardButton(text="check_progress"), KeyboardButton(text="stats"), KeyboardButton(text="recommendations"), KeyboardButton(text="reset_day"),],],
#         resize_keyboard=True,
#         one_time_keyboard=False
#     )


def after_set_profile_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="log_water"), KeyboardButton(text="log_food"), KeyboardButton(text="log_workout"), KeyboardButton(text="info"),
             ],],
        resize_keyboard=True,
        one_time_keyboard=False
    )

@dp.message(F.text == "set_profile")
async def btn_set_profile(message: types.Message, state: FSMContext):
    await set_profile(message, state)

@dp.message(F.text == "log_water")
async def btn_log_water(message: types.Message, state: FSMContext):
    await log_water(message, state)

@dp.message(F.text == "log_food")
async def btn_log_food(message: types.Message, state: FSMContext):
    await log_food(message, state)

@dp.message(F.text == "log_workout")
async def btn_log_workout(message: types.Message, state: FSMContext):
    await log_workout(message, state)

@dp.message(F.text == "info")
async def btn_info(message: types.Message):
    await info(message)

# Стартуем! Я сказала СТАРТУЕМ!
@dp.message(Command("start"))
async def start(message: types.Message, state: FSMContext):
    await message.answer(
        "Привлет helo 哈喽 👋! Я твой самый верный друг и помогу тебе следить за своим здоровьем.\n\n"
        "Команды:\n/set_profile - Настроить профиль (сделай это в первую очередь!)\n"
        "/log_water - Записать потребление воды\n/log_food - Записать потребление еды\n"
        "/log_workout - Записать информацию о тренировке\n/check_progress - Узнать потребление калорий и воды\n"
        "/stats - График прогресса по воде и калориям\n/recommendations - Получить рекомендации по питанию / тренировкам\n"
        "/reset_day - Начать новый день (тест)\n/who_am_i - Кто я, что я могу, и зачем я нужен?\n"
        "/info - Как пользоваться ботом (описание команд)\n/disclaimer - Дисклеймер (не полагайся полностью на бота!)\n"
        "Начни с команды /set_profile !",
        reply_markup=main_menu_keyboard()
    )

# добавим кнопку с информацией о том, как пользоваться ботом, чтобы пользователь не потерялся
@dp.message(Command("info"))
async def info(message: types.Message):
    await message.answer(
        "Как пользоваться ботом:\n\n"
        "Команды:\n/set_profile - Настроить профиль (самое важное!)\n"
        "/log_water - Записать потребление воды\n/log_food - Записать потребление еды\n"
        "/log_workout - Записать информацию о тренировке\n/check_progress - Узнать потребление калорий и воды\n"
        "/stats - График прогресса по воде и калориям\n/recommendations - Получить рекомендации по питанию / тренировкам\n"
        "/reset_day - Начать новый день (тест)\n/who_am_i - Кто я, что я могу, и зачем я нужен?\n"
        "/info - Как пользоваться ботом (описание команд)\n/disclaimer - Дисклеймер (не полагайся полностью на бота!)\n",
        reply_markup=after_set_profile_keyboard()
    )

# сэтапим профиль пользователя
@dp.message(Command("set_profile"))
async def set_profile(message: types.Message, state: FSMContext):
    await message.answer("Введи свой вес (кг):")
    await state.set_state(ProfileStates.weight)

@dp.message(ProfileStates.weight)
async def profile_weight(message: types.Message, state: FSMContext):
    try:
        w = float(message.text)
        if not validate_range(w, 20, 300):
            raise ValueError
        await state.update_data(weight=w)
        await message.answer("Введи свой рост (см):")
        await state.set_state(ProfileStates.height)
    except ValueError:
        await message.answer("Введи корректный вес (20–300 кг)!")

@dp.message(ProfileStates.height)
async def profile_height(message: types.Message, state: FSMContext):
    try:
        h = float(message.text)
        if not validate_range(h, 100, 250):
            raise ValueError
        await state.update_data(height=h)
        await message.answer("Введи возраст:")
        await state.set_state(ProfileStates.age)
    except ValueError:
        await message.answer("Введи корректный рост (100-250 см)!")

@dp.message(ProfileStates.age)
async def profile_age(message: types.Message, state: FSMContext):
    try:
        a = int(message.text)
        if not validate_range(a, 5, 100): # 100 лет это автору
            raise ValueError
        await state.update_data(age=a)
        await message.answer("Сколько в среднем минут в день ты уделяешь ВЫСОКОинтенсивной активности (бег / плавание / бокс, ...)?")
        await state.set_state(ProfileStates.high_intensity)
    except ValueError:
        await message.answer("Введи корректный возраст (5-100 лет)!")

@dp.message(ProfileStates.high_intensity)
async def profile_high_intensity(message: types.Message, state: FSMContext):
    try:
        hi = int(message.text)
        if hi < 0 or hi > 1440: # 24 x 60
            raise ValueError
        await state.update_data(high_minutes=hi)
        await message.answer("Сколько в среднем минут в день ты уделяешь НИЗКОинтенсивной активности (ходьба / йога / стретч)?")
        await state.set_state(ProfileStates.low_intensity)
    except ValueError:
        await message.answer("Введи корректное число минут (0–1440)!")

@dp.message(ProfileStates.low_intensity)
async def profile_low_intensity(message: types.Message, state: FSMContext):
    try:
        lo = int(message.text)
        if lo < 0 or lo > 1440:
            raise ValueError
        await state.update_data(low_minutes=lo)
        await message.answer("Твой город?")
        await state.set_state(ProfileStates.city)
    except ValueError:
        await message.answer("Введи корректное число минут (0–1440)!")

"""Здесь и далее используется примерно следующая логика:
Понятно, что условная ходьба и условный бег по-разному сжигают калории и влияют на норму воды
По-хорошему это надо разделять, но чтобы не усложнять логику слишком сильно, я разделил тренировки на высокоинтенсивные и низкоинтенсивные
У пользователя будет спрашиваться среднедневная активность по обоим типам, а также при логировании тренировки
Будет спрашиваться ее интенсивность. А дальше уже будет высчитываться все остальное)
"""
@dp.message(ProfileStates.city)
async def profile_city(message: types.Message, state: FSMContext):
    data = await state.get_data()
    city = message.text
    temp = await get_weather(city)
    weight = data["weight"]
    height = data["height"]
    age = data["age"]
    hi = data.get("high_minutes", 0)
    lo = data.get("low_minutes", 0)

    # пересчитываем нормы калорий и воды исходя из активности пользователя
    bmr = 10 * weight + 6.25 * height - 5 * age
    activity_kcal = hi * 8 + lo * 3
    calorie_norm = int(bmr + activity_kcal)
    base_water = int(weight * 30 + (hi // 30) * 300 + (lo // 30) * 150 + (500 if temp > 25 else 0))

    users[message.from_user.id] = {
        **data,
        "city": city,
        "base_water_norm": base_water,
        "daily_water_adjustment": 0,
        "water_norm": base_water,
        "calorie_norm": calorie_norm,
        "logged_water": 0,
        "logged_calories": 0,
        "burned_calories": 0,
        "today_high_minutes": 0,
        "today_low_minutes": 0,
        "last_reset": datetime.now().date()
    }

    await state.clear()
    # Формула для показа пользователю
    formula_text = (
        f"Формула для расчета калорий: 10 * weight + 6.25 * height - 5 * age + high_minutes * 8 + low_minutes * 3\n"
        f"Итог: {int(bmr)} + {int(activity_kcal)} = {calorie_norm} калорий в день\n\n"
        f"Формула для расчета воды: weight * 30 + (high_minutes / 30) * 300 + (low_minutes / 30) * 150."
        f"И еще + 500 мл, если в твоем городе жарко (температура выше 25 градусоы)\n"
        f"Итог: {base_water} мл воды в день"
    )

    # await message.answer(
    #     f"✅ Профиль сохранен\n"
    #     f"💧 Твоя дневная норма воды: {base_water} мл\n"
    #     f"🔥 Твоя дневная норма калорий: {calorie_norm} калорий\n\n"
    #     f"{formula_text}\n\n"
    #     "Теперь воспользуйся командами из меню для записи информации о потребленной воде, еде, о своих тренировках, или для получении статистики / советов!"
    # )
    await message.answer(
    f"✅ Профиль сохранен\n"
    f"💧 Твоя дневная норма воды: {base_water} мл\n"
    f"🔥 Твоя дневная норма калорий: {calorie_norm} калорий\n\n"
    f"{formula_text}\n\n"
    f"*Теперь воспользуйся командами из меню для записи информации о потребленной воде, еде, о своих тренировках, или для получении статистики / советов!*",
    parse_mode="Markdown",
    reply_markup=after_set_profile_keyboard()
)

@dp.message(Command("log_water"))
async def log_water(message: types.Message, state: FSMContext):
    uid = message.from_user.id
    if uid not in users:
        return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())

    args = message.text.split()
    if len(args) > 1:
        try:
            amount = int(args[1])
            users[uid]["logged_water"] += amount
            remaining = max(0, users[uid]["water_norm"] - users[uid]["logged_water"])
            await message.answer(f"💧 Записано {amount} мл\nОсталось до нормы: {remaining} мл", reply_markup=after_set_profile_keyboard())
        except ValueError:
            await message.answer("Введи число")
    else:
        await message.answer("Сколько мл воды было выпито?")
        await state.set_state(LogWaterStates.waiting_for_amount)

@dp.message(LogWaterStates.waiting_for_amount)
async def water_amount(message: types.Message, state: FSMContext):
    try:
        amount = int(message.text)
        users[message.from_user.id]["logged_water"] += amount
        remaining = max(0, users[message.from_user.id]["water_norm"] - users[message.from_user.id]["logged_water"])
        await message.answer(f"💧 Записано {amount} мл\nОсталось до нормы: {remaining} мл", reply_markup=after_set_profile_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Введи число")

@dp.message(Command("log_food"))
async def log_food(message: types.Message, state: FSMContext):
    uid = message.from_user.id
    if uid not in users:
        return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())

    args = message.text.split(maxsplit=2)
    if len(args) == 3 and args[2].isdigit():
        await state.update_data(product=args[1], amount=int(args[2]))
        await message.answer("В какой единице вводилось количество (выбери из кнопок ниже)?", reply_markup=unit_choice_keyboard())
        await state.set_state(FoodLogStates.waiting_for_unit)
    else:
        await message.answer("Введи название продукта:")
        await state.set_state(FoodLogStates.waiting_for_product)

@dp.message(FoodLogStates.waiting_for_product)
async def food_product(message: types.Message, state: FSMContext):
    await state.update_data(product=message.text)
    await message.answer("Сколько грамм / штук?")
    await state.set_state(FoodLogStates.waiting_for_amount)

@dp.message(FoodLogStates.waiting_for_amount)
async def food_amount(message: types.Message, state: FSMContext):
    try:
        amount = int(message.text)
        await state.update_data(amount=amount)
        await message.answer("В какой единице вводилось количество (выбери из кнопок ниже)?", reply_markup=unit_choice_keyboard())
        await state.set_state(FoodLogStates.waiting_for_unit)
    except ValueError:
        await message.answer("Введи число")

@dp.message(FoodLogStates.waiting_for_unit)
async def food_unit_choice(message: types.Message, state: FSMContext):
    text = message.text.strip().lower()
    data = await state.get_data()
    product = data.get("product")
    amount = data.get("amount")

    if not product or amount is None:
        await message.answer("Произошла ошибка, начни ввод продукта заново", reply_markup=ReplyKeyboardRemove())
        await state.clear()
        return

    if text == "граммы":
        await message.answer("Сейчас посчитаем...", reply_markup=ReplyKeyboardRemove())
        await process_food_entry(message, product, int(amount), state)
    elif text == "штуки":
        await message.answer("Оцениваю вес одной штуки с помощью LLM", reply_markup=ReplyKeyboardRemove())
        estimated = await estimate_piece_weight(product)
        if estimated is not None:
            total_grams = amount * estimated
            await message.answer(f"По оценке, одна штука ~{estimated} грамм. Итого: {total_grams} грамм")
            await process_food_entry(message, product, int(total_grams), state)
        else:
            await message.answer("Не удалось автоматически оценить вес одной штуки. Пожалуйста, введи примерный вес одной штуки в граммах:")
            await state.set_state(FoodLogStates.waiting_for_piece_grams)
    else:
        await message.answer("Пожалуйста, нажми кнопку 'грамм' или 'штук'")

@dp.message(FoodLogStates.waiting_for_piece_grams)
async def food_piece_weight(message: types.Message, state: FSMContext):
    try:
        grams_per_piece = int(message.text)
        data = await state.get_data()
        product = data.get("product")
        pieces = data.get("amount")
        if not product or pieces is None:
            await message.answer("Произошла ошибка — начни ввод продукта заново")
            await state.clear()
            return

        total_grams = pieces * grams_per_piece
        await message.answer("Ок, считаю...")
        await process_food_entry(message, product, int(total_grams), state)
    except ValueError:
        await message.answer("Введи число в граммах (например, 120)")

async def process_food_entry(message, product: str, amount: int, state: FSMContext):
    status = await message.answer("Считаю калории...")
    food = await get_food_info(product)

    if not food:
        return await status.edit_text("Не удалось определить продукт(")

    calories = food.calories_per_100g * amount / 100
    users[message.from_user.id]["logged_calories"] += calories

    await status.edit_text(
        f"🍽 {food.name}: {calories:.1f} калорий ({amount} г)"
    )

    await message.answer(
        "Добавить еще продукт?",
        reply_markup=more_food_keyboard()
    )
    await state.set_state(FoodLogStates.confirm_more)

@dp.message(FoodLogStates.confirm_more, F.text == "+ Добавить еще")
async def food_more(message: types.Message, state: FSMContext):
    await message.answer("Введи название продукта:", reply_markup=ReplyKeyboardRemove())
    await state.set_state(FoodLogStates.waiting_for_product)

@dp.message(FoodLogStates.confirm_more, F.text == "✅ Завершить")
async def food_finish(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Сохранил информацию о хрючеве!", reply_markup=after_set_profile_keyboard())


@dp.message(Command("log_workout"))
async def log_workout(message: types.Message, state: FSMContext):
    uid = message.from_user.id
    if uid not in users:
        return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())
    await message.answer("Выбери интенсивность тренировки (используй кнопки ниже):", reply_markup=intensity_keyboard())
    await state.set_state(WorkoutStates.choose_intensity)

@dp.message(WorkoutStates.choose_intensity)
async def workout_choose_intensity(message: types.Message, state: FSMContext):
    text = message.text.strip().lower()
    if text in ("высокая", "high", "высоко"):
        await state.update_data(intensity="high")
    elif text in ("низкая", "low", "низко"):
        await state.update_data(intensity="low")
    else:
        await message.answer("Пожалуйста, выбери 'Высокая' или 'Низкая'", reply_markup=intensity_keyboard())
        return
    await message.answer("Введи в формате: <тип тренировки> <минуты> (пример: бег 30)", reply_markup=ReplyKeyboardRemove())
    await state.set_state(WorkoutStates.waiting_for_input)

"""Тут логика -+ следующая:
у пользователя есть средние значения по нагрузкам типа high и low
если он эти значения по совокупности превысит - норма потребления воды конкретно для сегодня 
пересчитается и увеличится
"""
@dp.message(WorkoutStates.waiting_for_input)
async def workout_input(message: types.Message, state: FSMContext):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) == 2 and any(char.isdigit() for char in parts[1]):
            try:
                w_type, minutes_str = parts[0], parts[1]
                minutes = int(re.search(r"\d+", minutes_str).group(0))
            except Exception:
                w_type, minutes = message.text.split()
                minutes = int(minutes)
        else:
            w_type, minutes = message.text.split()
            minutes = int(minutes)

        uid = message.from_user.id
        if uid not in users:
            await state.clear()
            return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())

        weight = users[uid]["weight"]
        status_msg = await message.answer("Обрабатываю (как я задолбался кодить...)")

        # Детерминированная оценка сожженных калорий (сперва пробовал с LLM, но лучше не надо дядя)
        burned = estimate_burned_calories_det(w_type, minutes, weight)

        data = await state.get_data()
        chosen_intensity = data.get("intensity")
        # если пользователь выбрал интенсивность явно — используем ее, иначе классифицируем
        intensity = chosen_intensity if chosen_intensity in ("high", "low") else classify_workout_intensity(w_type)

        u = users[uid]
        if intensity == "high":
            u["today_high_minutes"] += minutes
        else:
            u["today_low_minutes"] += minutes
        avg_hi = u.get("high_minutes", 0)
        avg_lo = u.get("low_minutes", 0)

        excess_hi = max(0, u["today_high_minutes"] - avg_hi)
        excess_lo = max(0, u["today_low_minutes"] - avg_lo)

        # каждые 30 мин сверх среднего это +300 мл для high, +150 мл для low
        extra_high = (excess_hi // 30) * 300
        extra_low = (excess_lo // 30) * 150

        daily_adjustment = extra_high + extra_low
        u["daily_water_adjustment"] = daily_adjustment
        u["water_norm"] = u["base_water_norm"] + u["daily_water_adjustment"]
        u["burned_calories"] += burned

        await status_msg.edit_text(
            f"{w_type} {minutes} минут: сжег {burned} калорий (круто круто)\n"
            f"💧 По суммарному превышению средней нагрузки добавлено: {daily_adjustment} мл к норме воды\n"
            f"Текущая дневная норма воды: {u['water_norm']} мл\n\n"
            f"Суммарно сегодня: high {u['today_high_minutes']} минут, low {u['today_low_minutes']} минут\n\n"
            "Для дальнейшего запуска команд воспользуйся меню (три синие черточки в левом нижнем углу)"
        )
        await state.clear()
    except Exception:
        logging.exception("Workout input error")
        await message.answer("Неверный формат! Пример ввода: бег 30")
        await state.clear()


@dp.message(Command("check_progress"))
async def check_progress(message: types.Message):
    uid = message.from_user.id
    if uid not in users:
        return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())

    check_daily_reset(uid)
    u = users[uid]

    await message.answer(
        f"Статус:\n"
        f"💧 Вода: {u['logged_water']} / {u['water_norm']} мл\n"
        f"Калории: {u['logged_calories']:.1f} / {u['calorie_norm']} калорий\n"
        f"🔥 Сожжено: {u['burned_calories']} калорий\n\n"
        f"Сегодня: high {u.get('today_high_minutes',0)} минут, low {u.get('today_low_minutes',0)} минут\n"
        f"Средние: high {u.get('high_minutes',0)} минут, low {u.get('low_minutes',0)} минут",
        reply_markup=after_set_profile_keyboard()
    )

@dp.message(Command("stats"))
async def stats(message: types.Message):
    uid = message.from_user.id
    if uid not in users:
        return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())

    u = users[uid]

    # водичка
    plt.figure(figsize=(6, 4))
    labels = ["Вода (в ЛИТРАХ)"]
    target_water = [u["water_norm"] / 1000]
    current_water = [u["logged_water"] / 1000]

    x = range(len(labels))
    plt.bar(x, target_water, alpha=0.3, label="Цель")
    plt.bar(x, current_water, alpha=0.7, label="Факт")
    plt.xticks(x, labels)
    plt.legend()
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plt.close()

    # жрачка
    plt.figure(figsize=(6, 4))
    labels = ["Калории"]
    target_cal = [u["calorie_norm"]]
    current_cal = [u["logged_calories"]]

    x = range(len(labels))
    plt.bar(x, target_cal, alpha=0.3, label="Цель")
    plt.bar(x, current_cal, alpha=0.7, label="Факт")
    plt.xticks(x, labels)
    plt.legend()
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    plt.close()

    await message.answer_photo(
        BufferedInputFile(buf1.read(), filename="water_stats.png")
    )
    await message.answer_photo(
        BufferedInputFile(buf2.read(), filename="calories_stats.png")
    )

# рекомндации (путем направления запроса к LLM-агенту)
@dp.message(Command("recommendations"))
async def recommendations(message: types.Message):
    uid = message.from_user.id
    if uid not in users:
        return await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())

    u = users[uid]
    prompt = (
        f"Дай короткий совет по здоровью.\n"
        f"Вес: {u['weight']}\n"
        f"Вода: {u['logged_water']}/{u['water_norm']}\n"
        f"Калории: {u['logged_calories']}/{u['calorie_norm']}"
    )

    try:
        status = await message.answer("Думаю... скоро вернусь с результатами (hope nothing breaks again...)")
        res = await llm.ainvoke(prompt)
        await status.edit_text(f"Совет:\n{getattr(res,'content',str(res))}")
    except Exception:
        logging.exception("Ошибка с LLM (ну кто собственно сомневался)")
        await message.answer("Ошибка при получении рекомендаций(", reply_markup=after_set_profile_keyboard())

# сбросить значения по воде, калориям и активности (для тестирования)
@dp.message(Command("reset_day"))
async def reset_day(message: types.Message):
    uid = message.from_user.id
    if uid in users:
        users[uid].update({
            "logged_water": 0,
            "logged_calories": 0.0,
            "burned_calories": 0,
            "daily_water_adjustment": 0,
            "today_high_minutes": 0,
            "today_low_minutes": 0,
            "water_norm": users[uid].get("base_water_norm", 0),
            "last_reset": datetime.now().date()
        })
        await message.answer("🔄 День сброшен", reply_markup=after_set_profile_keyboard())
    else:
        await message.answer("Сначала /set_profile", reply_markup=main_menu_keyboard())


@dp.message(lambda message: message.text == "who_am_I")
async def who_am_i(message: types.Message):
    await message.answer(
        "Я — твой друг и помощник для отслеживания потребления воды, еды, и трекинга активности. "
        "Помогаю записывать выпитую воду, приемы пищи и тренировки, примерно оценивать калории и "
        "корректировать дневную норму воды в зависимости от активности. Помни, что я создан в образовательных целях и не заменяю консультации специалистов!",
        reply_markup=after_set_profile_keyboard()
    )

@dp.message(lambda message: message.text == "disclaimer")
async def disclaimer(message: types.Message):
    await message.answer(
        "Disclaimer: проект создан только в образовательных целях.\n\n"
        "Некоторые функции могут работать не точно (например, оценка калорий через LLM), и их доработка выходит за рамки данного pet-проекта.\n\n"
        "Пожалуйста, не полагайся полностью на мои советы и оценки, и в случае чего перепроверяй информацию с помощью надежных источников!"
        "",
        reply_markup=after_set_profile_keyboard()
    )

# регистрируем наши дисклеймеры и информацию о том, чем мы вообще занимаемся
@dp.message(Command("who_am_i"))
async def cmd_who_am_i(message: types.Message):
    await who_am_i(message)

@dp.message(Command("disclaimer"))
async def cmd_disclaimer(message: types.Message):
    await disclaimer(message)

# сколько можно уже писать давайте закускать!
async def main():
    await set_commands()
    logging.info("The miracle is about to begin...")
    await dp.start_polling(bot)

# локально-то запустить каждый может, а вот навернется ли это все на сервере - большой большой секрет
if __name__ == "__main__":
    asyncio.run(main())