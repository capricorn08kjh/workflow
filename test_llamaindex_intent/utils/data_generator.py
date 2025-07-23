import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
import json

class SampleDataGenerator:
    """풍부한 샘플 데이터 생성 클래스"""
    
    def __init__(self, db_path: str = "company_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def generate_all_data(self):
        """모든 샘플 데이터 생성"""
        print("샘플 데이터 생성 중...")
        
        # 정형 데이터
        self.create_performance_data()
        self.create_product_data()
        self.create_process_data()
        self.create_quality_data()
        self.create_safety_data()
        self.create_employee_data()
        
        # 비정형 문서 데이터
        self.create_meeting_minutes()
        self.create_monthly_reports()
        self.create_quality_reports()
        self.create_safety_reports()
        
        print("샘플 데이터 생성 완료!")
        
    def create_performance_data(self):
        """분기별 실적 데이터"""
        quarters = []
        revenues = []
        operating_profits = []
        companies = []
        
        base_revenue = 1000
        for year in [2022, 2023, 2024]:
            for q in range(1, 5):
                quarters.append(f"{year}Q{q}")
                # 약간의 성장과 계절성 반영
                seasonal_factor = 1.1 if q == 4 else (0.95 if q == 1 else 1.0)
                growth_factor = 1 + (year - 2022) * 0.15
                noise = random.uniform(0.9, 1.1)
                
                revenue = int(base_revenue * growth_factor * seasonal_factor * noise)
                revenues.append(revenue)
                operating_profits.append(int(revenue * random.uniform(0.08, 0.15)))
                companies.append('우리회사')
        
        df = pd.DataFrame({
            'quarter': quarters,
            'revenue': revenues,
            'operating_profit': operating_profits,
            'company': companies,
            'created_date': [datetime.now() - timedelta(days=random.randint(0, 1000)) for _ in range(len(quarters))]
        })
        
        df.to_sql('performance', self.conn, if_exists='replace', index=False)
        
    def create_product_data(self):
        """제품별 데이터"""
        products = ['스마트폰', '태블릿', '노트북', '스마트워치', '이어폰', '충전기']
        
        product_data = []
        for product in products:
            for month in range(1, 13):
                sales = random.randint(100, 500)
                market_share = random.uniform(10, 25)
                production_cost = sales * random.uniform(0.6, 0.8)
                
                product_data.append({
                    'product': product,
                    'month': f"2024-{month:02d}",
                    'sales': sales,
                    'market_share': market_share,
                    'production_cost': production_cost,
                    'units_sold': random.randint(1000, 5000)
                })
        
        pd.DataFrame(product_data).to_sql('product_sales', self.conn, if_exists='replace', index=False)
        
    def create_process_data(self):
        """공정별 실적 데이터"""
        processes = ['조립공정', '도장공정', '검사공정', '포장공정', '출하공정']
        
        process_data = []
        for process in processes:
            for day in range(1, 31):
                date = f"2024-01-{day:02d}"
                
                process_data.append({
                    'process_name': process,
                    'date': date,
                    'target_output': random.randint(800, 1200),
                    'actual_output': random.randint(700, 1100),
                    'efficiency': random.uniform(75, 95),
                    'defect_rate': random.uniform(0.5, 3.0),
                    'downtime_hours': random.uniform(0, 8),
                    'operator_count': random.randint(5, 15)
                })
        
        pd.DataFrame(process_data).to_sql('process_performance', self.conn, if_exists='replace', index=False)
        
    def create_quality_data(self):
        """품질 데이터"""
        quality_issues = ['외관불량', '기능불량', '치수불량', '조립불량', '재료불량']
        
        quality_data = []
        for i in range(200):
            issue_date = datetime.now() - timedelta(days=random.randint(0, 365))
            
            quality_data.append({
                'issue_id': f"Q{i+1:04d}",
                'issue_type': random.choice(quality_issues),
                'product': random.choice(['스마트폰', '태블릿', '노트북']),
                'severity': random.choice(['높음', '중간', '낮음']),
                'status': random.choice(['진행중', '완료', '보류']),
                'reported_date': issue_date.strftime('%Y-%m-%d'),
                'resolved_date': (issue_date + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d') if random.random() > 0.3 else None,
                'reporter': random.choice(['김영철', '이민수', '박서연', '정다은', '최준호'])
            })
        
        pd.DataFrame(quality_data).to_sql('quality_issues', self.conn, if_exists='replace', index=False)
        
    def create_safety_data(self):
        """안전사고 데이터"""
        accident_types = ['넘어짐', '베임', '끼임', '화상', '충돌']
        
        safety_data = []
        for i in range(50):
            accident_date = datetime.now() - timedelta(days=random.randint(0, 730))
            
            safety_data.append({
                'accident_id': f"S{i+1:04d}",
                'accident_type': random.choice(accident_types),
                'location': random.choice(['1공장', '2공장', '3공장', '창고', '사무실']),
                'severity': random.choice(['경상', '중상', '사망']),
                'injured_person': random.choice(['김영철', '이민수', '박서연', '정다은', '최준호', '강지민', '윤서현']),
                'accident_date': accident_date.strftime('%Y-%m-%d'),
                'cause': random.choice(['안전수칙 미준수', '장비 불량', '환경적 요인', '피로', '기타']),
                'prevention_action': random.choice(['안전교육 실시', '장비 교체', '작업환경 개선', '휴게시간 증가'])
            })
        
        pd.DataFrame(safety_data).to_sql('safety_accidents', self.conn, if_exists='replace', index=False)
        
    def create_employee_data(self):
        """직원 데이터"""
        employees = [
            '김영철', '이민수', '박서연', '정다은', '최준호', '강지민', '윤서현',
            '조성민', '한지우', '신예린', '배현우', '오다영', '임성호', '권미진'
        ]
        
        departments = ['생산팀', '품질팀', '안전팀', '기획팀', '영업팀', '연구개발팀']
        positions = ['사원', '대리', '과장', '차장', '부장']
        
        employee_data = []
        for emp in employees:
            employee_data.append({
                'name': emp,
                'employee_id': f"EMP{random.randint(1000, 9999)}",
                'department': random.choice(departments),
                'position': random.choice(positions),
                'hire_date': (datetime.now() - timedelta(days=random.randint(100, 3000))).strftime('%Y-%m-%d'),
                'salary': random.randint(3000, 8000),
                'email': f"{emp.lower()}@company.com"
            })
        
        pd.DataFrame(employee_data).to_sql('employees', self.conn, if_exists='replace', index=False)
    
    def create_meeting_minutes(self):
        """회의록 데이터 (JSON 파일로 저장)"""
        meeting_data = [
            {
                'meeting_id': 'M001',
                'title': '월간 생산 실적 검토 회의',
                'date': '2024-01-15',
                'attendees': ['김영철', '이민수', '박서연'],
                'content': """
                1. 1월 생산 실적 검토
                - 김영철 팀장: "이번 달 조립공정 효율이 85%로 목표치를 달성했습니다."
                - 이민수 과장: "하지만 도장공정에서 불량률이 2.5%로 다소 높습니다."
                
                2. 개선 방안 논의
                - 박서연 대리: "도장 설비 점검이 필요할 것 같습니다."
                - 김영철 팀장: "다음 주까지 점검 일정을 잡겠습니다."
                
                3. 다음 달 목표 설정
                - 전체 효율 90% 달성
                - 불량률 1.5% 이하 유지
                """,
                'keywords': ['생산실적', '효율', '불량률', '개선방안']
            },
            {
                'meeting_id': 'M002',
                'title': '품질 이슈 대응 회의',
                'date': '2024-01-20',
                'attendees': ['정다은', '김영철', '최준호'],
                'content': """
                1. 최근 품질 이슈 현황
                - 정다은 팀장: "스마트폰 외관불량이 증가하고 있습니다."
                - 김영철 팀장: "조립공정에서 발생하는 문제로 보입니다."
                
                2. 원인 분석
                - 최준호 과장: "작업자 교육 부족과 장비 노후화가 주요 원인입니다."
                
                3. 대응 방안
                - 작업자 재교육 실시
                - 장비 교체 검토
                - 김영철 팀장이 개선 계획 수립 담당
                """,
                'keywords': ['품질이슈', '외관불량', '작업자교육', '장비교체']
            },
            {
                'meeting_id': 'M003',
                'title': '안전 점검 결과 보고',
                'date': '2024-02-01',
                'attendees': ['강지민', '김영철', '윤서현'],
                'content': """
                1. 월간 안전점검 결과
                - 강지민 안전관리자: "1월 안전사고 2건 발생했습니다."
                - 윤서현 과장: "모두 경상 사고였지만 예방이 중요합니다."
                
                2. 안전 교육 계획
                - 김영철 팀장: "전 직원 대상 안전교육을 월 1회 실시하겠습니다."
                
                3. 안전설비 점검
                - 소화기 교체 필요
                - 비상구 표시등 수리 필요
                """,
                'keywords': ['안전점검', '안전사고', '안전교육', '안전설비']
            }
        ]
        
        # documents 폴더 생성
        Path("documents").mkdir(exist_ok=True)
        
        with open("documents/meeting_minutes.json", "w", encoding="utf-8") as f:
            json.dump(meeting_data, f, ensure_ascii=False, indent=2)
    
    def create_monthly_reports(self):
        """월간 보고서 데이터"""
        reports = [
            {
                'report_id': 'R001',
                'title': '2024년 4월 월간보고서',
                'date': '2024-04-30',
                'author': '이민수',
                'content': """
                ## 4월 월간 실적 요약
                
                ### 생산 실적
                - 총 생산량: 45,000대 (목표 대비 102%)
                - 스마트폰: 20,000대
                - 태블릿: 15,000대  
                - 노트북: 10,000대
                
                ### 품질 현황
                - 전체 불량률: 1.8% (목표 2.0%)
                - 고객 클레임: 15건 (전월 대비 -20%)
                
                ### 주요 이슈
                - 도장공정 설비 교체로 효율 개선
                - 김영철 팀장 주도로 작업자 교육 강화
                
                ### 다음 달 계획
                - 신제품 양산 준비
                - 품질 시스템 고도화
                """,
                'keywords': ['월간보고서', '생산실적', '품질현황', '4월']
            },
            {
                'report_id': 'R002',
                'title': '2024년 1분기 결산 실적',
                'date': '2024-03-31',
                'author': '박서연',
                'content': """
                ## 2024년 1분기 결산 실적
                
                ### 재무 성과
                - 매출액: 3,200억원 (전년 동기 대비 +12%)
                - 영업이익: 480억원 (영업이익률 15%)
                - 순이익: 360억원
                
                ### 사업부별 실적
                - 스마트폰 사업부: 1,800억원
                - 태블릿 사업부: 800억원
                - 노트북 사업부: 600억원
                
                ### 주요 성과
                - 신제품 출시 성공
                - 해외 시장 진출 확대
                - 김영철 팀장 등 핵심 인력 성과 우수
                
                ### 향후 전망
                - 2분기 매출 목표: 3,500억원
                - 신시장 개척 추진
                """,
                'keywords': ['결산실적', '1분기', '재무성과', '사업부별실적']
            }
        ]
        
        with open("documents/monthly_reports.json", "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
    
    def create_quality_reports(self):
        """품질 보고서 데이터"""
        quality_reports = [
            {
                'report_id': 'Q001',
                'title': '품질 이슈 종합 분석 보고서',
                'date': '2024-01-31',
                'author': '정다은',
                'content': """
                ## 1월 품질 이슈 분석
                
                ### 주요 품질 문제
                1. 외관불량 (40%)
                   - 스크래치, 얼룩 등
                   - 주요 원인: 포장 과정 미흡
                
                2. 기능불량 (30%)
                   - 터치 불량, 버튼 오동작
                   - 주요 원인: 조립 공정 문제
                
                3. 치수불량 (20%)
                   - 케이스 결합 불량
                   - 주요 원인: 금형 마모
                
                ### 반복 문제점
                - 작업자 숙련도 부족
                - 장비 노후화
                - 검사 기준 모호함
                - 김영철 팀장이 지적한 공정 간 연계 부족
                
                ### 개선 방안
                1. 작업자 재교육 프로그램 운영
                2. 노후 장비 교체 계획 수립
                3. 검사 기준 명확화
                4. 공정 간 소통 체계 개선
                """,
                'keywords': ['품질이슈', '반복문제', '개선방안', '외관불량']
            }
        ]
        
        with open("documents/quality_reports.json", "w", encoding="utf-8") as f:
            json.dump(quality_reports, f, ensure_ascii=False, indent=2)
    
    def create_safety_reports(self):
        """안전사고 보고서 데이터"""
        safety_reports = [
            {
                'report_id': 'S001',
                'title': '안전사고 분석 및 예방 대책',
                'date': '2024-02-15',
                'author': '강지민',
                'content': """
                ## 안전사고 현황 분석
                
                ### 사고 통계 (최근 6개월)
                - 총 사고 건수: 12건
                - 경상: 8건, 중상: 3건, 사망: 1건
                - 사고 유형: 넘어짐(4), 끼임(3), 베임(3), 기타(2)
                
                ### 반복되는 문제점
                1. 안전수칙 미준수
                   - 보호구 미착용 (50%)
                   - 작업 절차 무시 (30%)
                
                2. 환경적 요인
                   - 바닥 미끄러움
                   - 조명 부족
                   - 통로 적재물
                
                3. 인적 요인
                   - 안전 의식 부족
                   - 피로 누적
                   - 김영철 팀장 지적: 소통 부족
                
                ### 예방 대책
                1. 안전교육 강화
                2. 작업환경 개선
                3. 안전점검 횟수 증가
                4. 안전 인센티브 제도 도입
                """,
                'keywords': ['안전사고', '반복문제', '예방대책', '안전교육']
            }
        ]
        
        with open("documents/safety_reports.json", "w", encoding="utf-8") as f:
            json.dump(safety_reports, f, ensure_ascii=False, indent=2)
    
    def close(self):
        """데이터베이스 연결 종료"""
        self.conn.close()

def main():
    """샘플 데이터 생성 실행"""
    generator = SampleDataGenerator()
    generator.generate_all_data()
    generator.close()
    
    print("\n생성된 데이터:")
    print("- company_data.db: SQLite 데이터베이스")
    print("- documents/meeting_minutes.json: 회의록")
    print("- documents/monthly_reports.json: 월간 보고서")
    print("- documents/quality_reports.json: 품질 보고서")
    print("- documents/safety_reports.json: 안전사고 보고서")

if __name__ == "__main__":
    main()
