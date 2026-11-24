"""Export measured benchmark/mission scores to CSV and an offline HTML report."""
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

import evaluate_benchmarks as evaluator

OUT=evaluator.OUT


def export_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def number(x,digits=3):
    return '산출 불가' if x is None else f'{x:.{digits}f}'


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def plot_scores(hp,ud):
    # Visualization is separate from the OpenCV/NumPy evaluation implementation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Baseline','RootSIFT','MAGSAC++','RootSIFT\n+ MAGSAC++','RootSIFT\n+ MAGSAC++\n+ APAP']
    colors=['#8b97a5','#4e79a7','#779bc2','#45a29e','#167d6b']
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    settings=[(hp,'h_auc_5px_pct','HPatches: homography AUC @ 5 px','% (higher is better)'),
              (hp,'median_pair_point_median_px','HPatches: full-warp independent point error','px (lower is better)'),
              (ud,'mean_psnr_db','UDIS-D: common-overlap PSNR','dB (higher is better)'),
              (ud,'mean_ssim','UDIS-D: common-overlap SSIM','SSIM (higher is better)')]
    for ax,(data,key,title,ylabel) in zip(axes.ravel(),settings):
        if not data or all(r[key] is None for r in data):
            ax.set_title(title,fontsize=12)
            ax.text(.5,.5,'No scored pairs',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
            continue
        values=[r[key] if r[key] is not None else float('nan') for r in data]
        bars=ax.bar(labels,values,color=colors)
        ax.set_title(title,fontsize=12);ax.set_ylabel(ylabel)
        ax.spines[['top','right']].set_visible(False)
        finite=[v for v in values if math.isfinite(v)]
        ax.set_ylim(0,max(finite)*1.22 if finite and max(finite)>0 else 1)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for bar,value in zip(bars,values):
            if not math.isfinite(value):continue
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+ax.get_ylim()[1]*.018,
                    f'{value:.3f}',ha='center',fontsize=10)
    fig.suptitle(f'Pairwise pipelines: {hp[0]["pairs"] if hp else 0} HPatches pairs / {ud[0]["pairs"] if ud else 0} UDIS-D pairs',fontsize=14)
    fig.text(.5,.012,'Point-error medians use successful fits; AUC includes failures. No exposure correction or blending.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.95])
    fig.savefig(OUT/'benchmark_scores.png',dpi=180)
    fig.savefig(OUT/'benchmark_scores.pdf')
    plt.close(fig)


def main():
    rows=[json.loads(line) for line in (OUT/'benchmark_pairs.jsonl').read_text().splitlines() if line.strip()]
    protocol=json.loads((OUT/'benchmark_protocol.json').read_text())
    integrity=json.loads((OUT/'dataset_integrity.json').read_text())
    summary=evaluator.clean_json(evaluator.summarize(rows))
    (OUT/'benchmark_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    export_csv(OUT/'benchmark_summary.csv',summary)
    detail=[]
    for row in rows:
        for method in row['methods']:
            detail.append(dict(dataset=row['dataset'],pair=row['id'],group=row.get('group'),
                               **{k:v for k,v in method.items() if not isinstance(v,(dict,list))}))
    export_csv(OUT/'benchmark_pairs.csv',detail)
    groups=[]
    for group in ['i','v']:
        for record in evaluator.summarize([r for r in rows if r['dataset']=='hpatches' and r['group']==group]):
            groups.append(dict(group=group,**record))
    export_csv(OUT/'hpatches_groups.csv',groups)
    missions=json.loads((OUT/'mission_photometric.json').read_text()) if (OUT/'mission_photometric.json').exists() else []
    export_csv(OUT/'mission_summary.csv',[dict(method=r['method'],mission=r['mission'],
        overlap_pairs=len(r['common_pair_ids']),psnr_db=r['mean_overlap_psnr_db'],ssim=r['mean_overlap_ssim'],
        independent_point_error=None,corner_error=None) for r in missions])
    hp=[r for r in summary if r['dataset']=='hpatches']
    ud=[r for r in summary if r['dataset']=='udis_d']
    plot_scores(hp,ud)
    paragraphs=['<!doctype html><html lang="ko"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>이미지 정합 정량 평가</title>',
        '<style>body{font:16px/1.7 system-ui;color:#172536;max-width:1500px;margin:36px auto;padding:0 24px}h1,h2{line-height:1.4}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border:1px solid #cbd5e1;padding:9px;text-align:left}th{background:#edf3f8}a{color:#075cab}.note{background:#eef5ff;padding:16px}img{max-width:100%}code{background:#edf3f8} @media print{body{font-size:10pt}table{font-size:9pt}tr{break-inside:avoid}}</style>',
        '<h1>다섯 방법의 정량 평가</h1>',
        f'<p>이번 실행: HPatches {sum(r["dataset"]=="hpatches" for r in rows)}쌍, UDIS-D {sum(r["dataset"]=="udis_d" for r in rows)}쌍. 총 {len(rows)}쌍 × 5개 방법 = {len(detail):,}회 평가.</p>',
        '<p class="note">이 벤치마크는 저장된 각 방법의 <strong>Q1 두 영상 정합 경로</strong>를 그대로 실행하여 매칭·호모그래피·APAP 워핑을 평가했다. 8장 카메라 BA 또는 최종 블렌딩 품질의 벤치마크 점수는 아니다. 정답 변환은 평가에만 사용했다.</p>',
        '<p><a href="benchmark_summary.csv">요약 CSV</a> · <a href="benchmark_pairs.csv">모든 쌍의 CSV</a> · <a href="benchmark_protocol.json">평가 조건·입력 해시</a> · <a href="dataset_integrity.json">데이터 무결성·누락 기록</a> · <a href="../comparison.html">정합 결과 이미지</a></p>',
        '<p><a href="benchmark_scores.png">발표용 그래프 PNG</a> · <a href="benchmark_scores.pdf">그래프 PDF</a></p><img src="benchmark_scores.png" alt="벤치마크 점수 비교 그래프">',
        '<h2>HPatches: 정답 기반 기하 정확도</h2>',
        '<p>독립 평가점은 영상당 20×20 격자 중 정답 변환으로 상대 영상 내부에 들어오는 점이다. 모델 추정이나 APAP 제어점으로 쓰지 않았다. 위치 오차는 상대 영상 원본 픽셀 단위다. 독립점 중앙값·P95는 쌍별 값들의 중앙값, RMSE는 쌍별 RMSE의 평균이다. 오차 요약은 성공 사례 기준이며 실패를 포함한 성능은 AUC·성공률로 함께 제시한다.</p>',
        table(['방법','성공/전체','독립점 중앙값(px) ↓','독립점 RMSE(px) ↓','독립점 P95(px) ↓','H corner 중앙값(px) ↓','H 성공@3px(%) ↑','H AUC@5px(%) ↑'],[
            [r['method'],f'{r["successful"]}/{r["pairs"]}',number(r['median_pair_point_median_px']),number(r['mean_point_rmse_px']),number(r['median_pair_point_p95_px']),number(r['median_pair_corner_error_px']),number(r['h_success_3px_pct'],2),number(r['h_auc_5px_pct'],2)] for r in hp]),
        '<p>H corner는 APAP 이전 전역 호모그래피를 평가한다. 따라서 RootSIFT+MAGSAC++와 APAP 확장판의 H 점수는 같을 수 있으며, APAP의 추가 효과는 독립점의 전체 워핑 오차로 확인한다. H AUC@t는 100×mean(max(1−error/t,0))이며 실패의 오차는 무한대로 두어 0점 처리한다.</p>',
        table(['방법','H AUC@3px(%)','H AUC@5px(%)','H AUC@10px(%)','PSNR(dB) ↑','SSIM ↑'],[
            [r['method'],number(r['h_auc_3px_pct'],2),number(r['h_auc_5px_pct'],2),number(r['h_auc_10px_pct'],2),number(r['mean_psnr_db']),number(r['mean_ssim'],4)] for r in hp]),
        '<p>HPatches 광도 점수는 모든 방법에 동일한 정답 겹침 영역을 사용한다. 추정 워핑이 덮지 못한 부분은 검정으로 남겨 오차에 반영한다. 조명 변화 자체도 PSNR·SSIM을 낮추므로 이 수치를 기하 정확도로 해석하지 않는다. 원본 좌표를 사용하는 자체 평가 프로토콜이며 다른 논문의 subset·해상도·AUC 구현과 직접 수치 비교하지 않는다.</p>',
        '<h2>UDIS-D: 겹침 영역의 광도 일치도</h2>',
        '<p>주지표는 같은 영상 쌍에서 다섯 방법의 유효 영역이 공통으로 겹치는 곳에 대한 PSNR·SSIM이다. 노출 보정과 블렌딩을 적용하지 않았다. 이 데이터에는 정답 호모그래피/독립 랜드마크가 없으므로 corner error와 참 위치 오차는 산출하지 않는다.</p>',
        table(['방법','정합 성공/전체','공통 영역 점수 산출 쌍','PSNR(dB) ↑','SSIM ↑','평균 평가 면적(%)'],[
            [r['method'],f'{r["successful"]}/{r["pairs"]}',r['n_psnr_db'],number(r['mean_psnr_db']),number(r['mean_ssim'],4),number(r['mean_scored_area_fraction']*100 if r['mean_scored_area_fraction'] is not None else None,2)] for r in ud]),
        '<p>공통 영역은 모든 방법의 추정 결과에 영향을 받으므로, 면적 및 개별 겹침 점수도 함께 보관했다. SSIM의 완전히 유효한 11×11 윈도가 256개 미만인 쌍은 공통 영역 점수를 산출하지 않는다. 정합 실패는 별도로 집계하고, 광도 비교에서는 항등변환으로 대체한다.</p>',
        '<h2>UDIS-D 보조 점수: 저자 코드와 유사한 masked-canvas 방식</h2>',
        table(['방법','개별 overlap PSNR','개별 overlap SSIM','Masked-canvas PSNR','Masked-canvas SSIM'],[
            [r['method'],number(r['mean_own_overlap_psnr_db']),number(r['mean_own_overlap_ssim'],4),number(r['mean_masked_canvas_psnr_db']),number(r['mean_masked_canvas_ssim'],4)] for r in ud]),
        '<p>저자 코드는 두 영상에 마스크를 곱한 뒤 전체 캔버스를 평균한다. 여기서는 같은 원리를 OpenCV/NumPy로 구현하고 SSIM은 7×7 uniform window·sample covariance로 계산했다. 검정 패딩이 많으면 점수가 높아질 수 있어 보조 수치로만 사용한다. OpenCV 보간·이진 마스크를 사용하므로 저자의 PyTorch warp와 비트 단위로 동일한 공식 재현은 아니다. <a href="https://github.com/nie-lang/UDIS2/blob/main/Warp/Codes/test.py">저자 평가 코드</a></p>',
        '<h2>프로젝트 Q1·Q2·Q3·E1·E2: 후처리 전 겹침 영역 진단</h2>',
        '<p>아래 수치는 최종 PNG끼리 비교한 점수가 아니라 각 입력을 워핑한 직후의 중첩 영상 비교다. 방법별로 공통으로 평가 가능한 영상 쌍 목록을 고정했지만 픽셀 겹침 영역·출력 축척은 각 방법의 추정값이므로, 정답 기반의 공정한 정확도 순위로 사용하지 않는다. Q1/Q2/Q3/E1/E2 사이의 점수도 직접 순위 비교하지 않는다.</p>',
        table(['방법','미션','평가 겹침 쌍','PSNR(dB)','SSIM','독립점·corner 정확도'],[
            [r['method'],r['mission'],len(r['common_pair_ids']),number(r['mean_overlap_psnr_db']),number(r['mean_overlap_ssim'],4),'정답 없음'] for r in missions]),
        '<p>E1/E2는 같은 기하 결과이며 이 표는 노출 보정·블렌딩 전을 측정하므로 점수가 같다. E2의 외관 개선은 결과 이미지의 동일 영역 확대 비교로 제시한다.</p>',
        '<h2>데이터·실패·재현 조건</h2>',
        f'<p>로컬 UDIS-D input1 {integrity["udis_input1_count"]}장, input2 {integrity["udis_input2_count"]}장 중 같은 파일명의 완전한 쌍은 {integrity["udis_complete_pairs"]}개다. 선택·처리한 쌍 수는 위 실행 요약을 따른다. 누락 목록과 배포본 참조 해시는 dataset_integrity.json에 기록한다.</p>',
        '<p>HPatches 실패도 이번 실행에서 평가한 전체 쌍의 분모에 포함했다. 오차 중앙값만 비교하면 실패가 많은 방법에 유리할 수 있으므로 반드시 성공 수와 AUC를 함께 읽어야 한다. PSNR은 0오차에서 100dB로 제한했다. 기본 SSIM은 BGR 채널 평균, 11×11 Gaussian(σ=1.5), K1=.01/K2=.03, data_range=255, population covariance를 사용한다.</p>',
        '<p>기존 brown+lowe의 다중영상 BA는 OpenCV이고 나머지는 NumPy LM이지만, 공개 벤치마크에 사용한 Q1 경로에는 BA가 없어 이 차이가 벤치마크 점수에 개입하지 않는다. 프로젝트 Q2/Q3/E1/E2 진단에는 이 차이가 포함된다. APAP는 현재 프로젝트의 제약을 넣은 Moving DLT 확장판이다.</p>',
        '<p>CSV의 seconds는 병렬 작업 중 측정한 1회 기하 처리 시간으로, 입력은 메모리에 미리 읽어 두었고 광도 지표 계산 시간은 제외된다. 반복 측정한 단독 실행 속도 벤치마크로 사용하지 않는다.</p>',
        '<pre>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B experiments/evaluate_benchmarks.py --workers 4\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B experiments/evaluate_missions.py\npython3 -B experiments/report_evaluation.py</pre>',
        '<p><a href="https://github.com/hpatches/hpatches-dataset">HPatches 공식 배포처</a> · <a href="https://github.com/nie-lang/UnsupervisedDeepImageStitching">UDIS-D 공식 배포처</a> · <a href="https://ece.uwaterloo.ca/~z70wang/research/ssim/">SSIM 원 저자 설명</a></p>',
        '</html>']
    (OUT/'report.html').write_text('\n'.join(paragraphs),encoding='utf-8')
    print('Report and CSV exports written:',OUT)
    for r in hp:
        print('HP',r['method'],r['successful'],'median',r['median_pair_point_median_px'],'AUC5',r['h_auc_5px_pct'])
    for r in ud:
        print('UDIS',r['method'],r['successful'],'PSNR',r['mean_psnr_db'],'SSIM',r['mean_ssim'])


if __name__=='__main__':main()
