from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import re
import os

class M2M100Translator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = 'facebook/m2m100_418M'
        
        print(f"Загрузка модели на устройство: {self.device}")
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Перенос модели на GPU
        self.model = self.model.to(self.device)
        
    def translate_single(self, text, src_lang="en", tgt_lang="ru"):
        """Перевод одного текста с указанием языков"""
        try:
            # Устанавливаем исходный язык
            self.tokenizer.src_lang = src_lang
            
            # Токенизация с переносом на нужное устройство
            input_ids = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Генерация с указанием целевого языка
            generated_tokens = self.model.generate(
                **input_ids,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                num_beams=5,
                early_stopping=True,
                max_length=512
            )
            
            # Перенос обратно на CPU для декодирования
            result = self.tokenizer.decode(
                generated_tokens[0].cpu(), 
                skip_special_tokens=True
            )
            return result
            
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            return "ERROR_TRANSLATION"

def translate_file_deep(file_path, output_path):
    translator = M2M100Translator()
    
    # Регулярное выражение для извлечения компонентов
    pattern = r'^(\S+)\s+(\S+)\s+\[([^\]]+)\].*$'
    
    # Создаем или очищаем выходной файл
    open(output_path, 'w', encoding='utf-8').close()
    
    # Читаем весь файл
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Обрабатываем строки с записью после каждой обработки
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            # Записываем пустую строку
            with open(output_path, 'a', encoding='utf-8') as f_out:
                f_out.write('\n')
            continue
            
        match = re.match(pattern, line)
        if match:
            hanzi_trad, hanzi_simp, pinyin = match.groups()
            
            # Переводим каждую строку отдельно (предполагаем, что исходный текст - китайский)
            translation = translator.translate_single(hanzi_simp, src_lang="zh", tgt_lang="ru")
            new_line = f"{hanzi_trad} {hanzi_simp} [{pinyin}] /{translation}/"
            
            # Записываем результат сразу после обработки
            with open(output_path, 'a', encoding='utf-8') as f_out:
                f_out.write(new_line + '\n')
            
            # Вывод прогресса
            if i % 100 == 0:
                print(f"Обработано {i}/{len(lines)} строк...")
                # Периодическая очистка памяти GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Записываем строки, не подходящие под шаблон
            with open(output_path, 'a', encoding='utf-8') as f_out:
                f_out.write(line + '\n')
    
    print(f"Перевод завершен! Результат сохранен в {output_path}")

# Дополнительная функция для очистки памяти
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Память GPU очищена")

# Использование
if __name__ == "__main__":
    # Проверка доступности GPU
    if torch.cuda.is_available():
        print(f"Используется GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU не обнаружен, используется CPU")
    
    translate_file_deep('hu.txt', 'ru.txt')
    clear_gpu_memory()