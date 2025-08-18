import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import fitz  # PyMuPDF
import re
from typing import List, Dict
import json

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text_from_pdf(self) -> str:
        """PDF에서 텍스트를 추출합니다."""
        text = ""
        try:
            doc = fitz.open(self.pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            print(f"PDF 읽기 오류: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """텍스트를 정리합니다."""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?;:()]', '', text)
        return text.strip()
    
    def split_into_smart_chunks(self, text: str, target_chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """문장 단위로 청크를 나누어 문맥을 보존합니다."""
        # 문장 단위로 분할 (마침표, 느낌표, 물음표 기준)
        sentence_parts = re.split(r'([.!?]+)', text)
        
        # 문장과 구두점을 다시 결합
        sentences = []
        for i in range(0, len(sentence_parts), 2):
            if i + 1 < len(sentence_parts):
                sentences.append(sentence_parts[i] + sentence_parts[i + 1])
            else:
                sentences.append(sentence_parts[i])
        
        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 현재 청크에 문장을 추가했을 때의 길이
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= target_chunk_size:
                # 청크 크기 내에 있으면 문장 추가
                current_chunk = potential_chunk
            else:
                # 청크 크기를 초과하면 현재 청크 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 새 청크 시작 (오버랩 고려)
                if chunks and overlap > 0:
                    # 이전 청크의 끝부분을 오버랩으로 사용
                    overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class FinetuningDataProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_qa_prompts(self, text_chunks: List[str]) -> List[Dict]:
        """텍스트 청크에서 QA 쌍을 생성합니다."""
        qa_data = []
        
        # 사이버 보안 관련 질문 템플릿들
        question_templates = [
            "이 문서에서 언급된 주요 사이버 위협은 무엇인가요?",
            "2024년 하반기 사이버 공격의 특징은 무엇인가요?",
            "기업들이 취해야 할 사이버 보안 대응 방안은 무엇인가요?",
            "최근 사이버 공격의 주요 대상은 무엇인가요?",
            "사이버 보안 위험을 줄이기 위한 권장사항은 무엇인가요?",
            "이 문서에서 제시하는 보안 정책의 핵심 내용은 무엇인가요?",
            "사이버 공격의 새로운 트렌드는 무엇인가요?",
            "조직의 보안 인식 제고를 위한 방안은 무엇인가요?",
            "이 문서에서 언급된 보안 기술의 발전 방향은 무엇인가요?",
            "사이버 보안 사고 대응 절차는 어떻게 되어있나요?"
        ]
        
        for chunk in text_chunks:
            # 각 청크에 대해 여러 질문 생성
            for question in question_templates:
                # 간단한 키워드 매칭으로 관련성 확인
                if self._is_relevant_chunk(chunk, question):
                    answer = self._generate_answer_from_context(chunk, question)
                    qa_data.append({
                        "question": question,
                        "context": chunk,
                        "answer": answer
                    })
        
        return qa_data
    
    def _generate_answer_from_context(self, context: str, question: str) -> str:
        """컨텍스트와 질문을 바탕으로 답변을 생성합니다."""
        # 질문 유형에 따른 답변 템플릿
        if "위협" in question:
            return f"주어진 문서에 따르면, 주요 사이버 위협으로는 {self._extract_keywords(context, ['공격', '위협', '해킹', '침해'])} 등이 있습니다."
        elif "특징" in question:
            return f"2024년 하반기 사이버 공격의 특징으로는 {self._extract_keywords(context, ['새로운', '발전', '변화', '트렌드'])} 등이 나타나고 있습니다."
        elif "대응" in question or "방안" in question:
            return f"기업들이 취해야 할 사이버 보안 대응 방안으로는 {self._extract_keywords(context, ['대응', '방안', '정책', '절차'])} 등이 있습니다."
        elif "대상" in question:
            return f"최근 사이버 공격의 주요 대상으로는 {self._extract_keywords(context, ['기업', '조직', '시스템', '인프라'])} 등이 있습니다."
        elif "권장사항" in question:
            return f"사이버 보안 위험을 줄이기 위한 권장사항으로는 {self._extract_keywords(context, ['보안', '안전', '보호', '방어'])} 등이 있습니다."
        else:
            return f"주어진 문서에 따르면, {context[:150]}... (문서 내용을 바탕으로 한 답변)"
    
    def _extract_keywords(self, text: str, keywords: List[str]) -> str:
        """텍스트에서 키워드를 추출합니다."""
        found_keywords = []
        for keyword in keywords:
            if keyword in text:
                found_keywords.append(keyword)
        return ", ".join(found_keywords) if found_keywords else "다양한 보안 요소"
    
    def _is_relevant_chunk(self, chunk: str, question: str) -> bool:
        """청크가 질문과 관련이 있는지 확인합니다."""
        # 간단한 키워드 매칭
        keywords = {
            "위협": ["위협", "공격", "해킹", "침해"],
            "대응": ["대응", "방안", "정책", "절차"],
            "기술": ["기술", "시스템", "플랫폼", "솔루션"],
            "트렌드": ["트렌드", "동향", "변화", "발전"],
            "보안": ["보안", "안전", "보호", "방어"]
        }
        
        for category, words in keywords.items():
            if any(word in question for word in words):
                if any(word in chunk for word in words):
                    return True
        return True  # 기본적으로 모든 청크를 포함
    
    def create_training_data(self, text_chunks: List[str]) -> Dataset:
        """텍스트 청크를 QA 훈련 데이터로 변환합니다."""
        qa_data = self.create_qa_prompts(text_chunks)
        formatted_data = []
        
        for qa in qa_data:
            # 프롬프트 템플릿 (일반 System/User/Assistant 형식)
            formatted_text = f"""### System:
당신은 사이버 보안 전문가입니다. 주어진 문서를 바탕으로 질문에 답변해주세요.

### User:
문서: {qa['context']}

질문: {qa['question']}

### Assistant:
{qa['answer']}"""
            
            formatted_data.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_function(self, examples):
        """텍스트를 토큰화합니다."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

def load_model_for_finetuning(model_name: str):
    """파인튜닝을 위한 모델을 로드합니다."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model

def main():
    # 설정
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    PDF_PATH = "minyoung/finetuning/data/pdfs/2024 하반기 사이버 위협 동향 보고서.pdf"
    OUTPUT_DIR = "minyoung/finetuning/output"
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. PDF 텍스트 추출 중...")
    pdf_processor = PDFProcessor(PDF_PATH)
    raw_text = pdf_processor.extract_text_from_pdf()
    cleaned_text = pdf_processor.clean_text(raw_text)
    text_chunks = pdf_processor.split_into_smart_chunks(cleaned_text, target_chunk_size=500, overlap=50)
    
    print(f"추출된 텍스트 청크 수: {len(text_chunks)}")
    
    print("2. 훈련 데이터 준비 중...")
    data_processor = FinetuningDataProcessor(MODEL_NAME)
    dataset = data_processor.create_training_data(text_chunks)
    
    # 데이터셋을 train/validation으로 분할
    dataset = dataset.train_test_split(test_size=0.1)
    
    print("3. 모델 로드 중...")
    model = load_model_for_finetuning(MODEL_NAME)
    
    print("4. 훈련 설정 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=4,
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=data_processor.tokenizer,
        mlm=False,
    )
    
    print("5. 훈련 시작...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=data_processor.tokenizer,
    )
    
    # 훈련 실행
    trainer.train()
    
    print("6. 모델 저장 중...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    data_processor.tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print("파인튜닝 완료!")
    print(f"모델이 {OUTPUT_DIR}/final_model에 저장되었습니다.")

if __name__ == "__main__":
    main() 
