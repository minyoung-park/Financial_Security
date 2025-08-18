import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_finetuned_model(model_path: str):
    """파인튜닝된 모델을 로드합니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    return pipe

def test_model(pipe, test_prompts: list):
    """모델을 테스트합니다."""
    print("=== 파인튜닝된 모델 테스트 ===\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"테스트 {i}:")
        print(f"입력: {prompt}")
        print("-" * 50)
        
        response = pipe(prompt)[0]['generated_text']
        print(f"출력: {response}")
        print("=" * 80 + "\n")

def main():
    # 파인튜닝된 모델 경로
    FINETUNED_MODEL_PATH = "minyoung/finetuning/output/final_model"
    
    # 테스트 프롬프트들 (QA 형식)
    test_prompts = [
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 사이버 보안 전문가입니다. 주어진 문서를 바탕으로 질문에 답변해주세요.

<|eot_id|><|start_header_id|>user<|end_header_id|>

문서: 2024년 하반기 사이버 위협 동향 보고서의 주요 내용

질문: 이 문서에서 언급된 주요 사이버 위협은 무엇인가요?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 사이버 보안 전문가입니다. 주어진 문서를 바탕으로 질문에 답변해주세요.

<|eot_id|><|start_header_id|>user<|end_header_id|>

문서: 2024년 하반기 사이버 위협 동향 보고서의 주요 내용

질문: 2024년 하반기 사이버 공격의 특징은 무엇인가요?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 사이버 보안 전문가입니다. 주어진 문서를 바탕으로 질문에 답변해주세요.

<|eot_id|><|start_header_id|>user<|end_header_id|>

문서: 2024년 하반기 사이버 위협 동향 보고서의 주요 내용

질문: 기업들이 취해야 할 사이버 보안 대응 방안은 무엇인가요?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    ]
    
    try:
        print("파인튜닝된 모델 로드 중...")
        pipe = load_finetuned_model(FINETUNED_MODEL_PATH)
        print("모델 로드 완료!")
        
        test_model(pipe, test_prompts)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("파인튜닝이 완료되지 않았거나 모델 경로를 확인해주세요.")

if __name__ == "__main__":
    main()
