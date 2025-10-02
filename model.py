
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import asyncio
import time
from asyncio import Semaphore
from dataclasses import dataclass
@dataclass
class Prediction:
    start_index: int
    end_index: int
    entity: str




class Observer:
    def __init__(self):
        self._event = asyncio.Event()
        self.request: str | None = None
        self.response: list  | None = None
        self.status = self._event.wait()

    def update(self):
        self._event.set()


class TimerAwait:
    @staticmethod
    async def timer_with_await(func, milliseconds: int):
        if func is None:
            raise ValueError("func must not be None")
        if milliseconds <= 0:
            raise ValueError("milliseconds must be positive")

        while True:
            start_time = time.time() * 1000  # ms
            await func()
            end_time = time.time() * 1000
            delay = milliseconds - (end_time - start_time)
            if delay > 0:
                await asyncio.sleep(delay / 1000)


class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._observers: list[Observer] = []
        self._lock = asyncio.Semaphore(1)
        self.model_name = 'models/FacebookAI/xlm-roberta-large'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name )
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

    async def start(self):
        asyncio.create_task(TimerAwait.timer_with_await(self.predict, 200))

    async def get_request(self, text: str) -> str:
        observer = await self.wait_until_update(text)
        await observer.status
        return observer.response

    async def wait_until_update(self, text: str) -> Observer:
        async with self._lock:
            observer = Observer()
            observer.request = text
            self._observers.append(observer)
            return observer

    async def predict(self):
        async with self._lock:
            samples = []
            for observer in self._observers:
                samples.append(observer.request.split())

            if len(samples) > 0:
             preprocessed_samples = self.preprocess(samples)
             predictions = self.infer(preprocessed_samples['inputs'])
             results = self.postprocess(samples,preprocessed_samples['batch_word_indexes'],predictions)
             for observer,result in zip(self._observers,results):
                observer.response = result
            
            for observer in self._observers:
                observer.update()
            self._observers.clear()

    def preprocess(self,samples):
         batch_word_indexes = []
         inputs = self.tokenizer(samples, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)  
         for i in range(len(samples)):
              word_ids = inputs.word_ids(batch_index=i)
              previous_word_idx = None
              words_indexes = []
              for i, word_idx in enumerate(word_ids): 
                 if  word_idx != previous_word_idx: 
                    words_indexes.append(i)
                 previous_word_idx = word_idx
              batch_word_indexes.append(words_indexes) 

         return {'inputs' : inputs, 'batch_word_indexes' : batch_word_indexes}
    
    def infer(self,inputs):
         with torch.no_grad():
              logits = self.model(input_ids=inputs['input_ids'].to(self.device),attention_mask=inputs['attention_mask'].to(self.device)).logits
         return torch.argmax(logits, dim=2)
    
    def postprocess(self,samples,batch_words_indexes,batch_predictions):
        batch_final_estimation = []
        for words,words_indexes,predictions in zip(samples,batch_words_indexes,batch_predictions):
         predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions]
         final_estimation = []
         previous_word_begin = words_indexes[0]
         gap_counter = 0
         previous_letter_idx = 0
         letter_idx = len(words[0])
         for i, word_begin in enumerate(words_indexes[1:]):
            final_estimation.append(Prediction(start_index=previous_letter_idx + gap_counter,end_index=letter_idx + gap_counter,entity=predicted_token_class[previous_word_begin]))
            gap_counter += 1
            previous_word_begin = word_begin
            previous_letter_idx = letter_idx
            if i != len(words_indexes[1:]) - 1:
                letter_idx += len(words[i + 1])
         batch_final_estimation.append(final_estimation)  
        return batch_final_estimation
      
    


 
    
        





    
    

