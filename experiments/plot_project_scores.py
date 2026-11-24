"""Make presentation figures from measured final-PNG scores, without rerunning stitching."""
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='project-eval-mpl-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_project_results import OUT, METHODS, MISSIONS


def main():
    rows = json.loads((OUT/'results.json').read_text())
    colors = ['#4e79a7','#779bc2','#45a29e','#167d6b']
    labels = ['RootSIFT','MAGSAC++','RootSIFT + MAGSAC++','RootSIFT + MAGSAC++ + APAP']
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    x = np.arange(5);width = .19
    for ax, key, title in zip(axes,
            ['source_consistency_psnr_db','source_consistency_ssim'],
            ['PSNR (dB): saved panorama vs original inputs', 'SSIM: saved panorama vs original inputs']):
        for i, folder in enumerate(METHODS):
            values = [next(r[key] for r in rows if r['method']==folder and r['mission']==m) for m in MISSIONS]
            bars = ax.bar(x+(i-1.5)*width, values, width, label=labels[i], color=colors[i])
            ax.bar_label(bars, fmt='%.2f' if key.endswith('db') else '%.3f', padding=3, fontsize=8)
        ax.set_xticks(x, MISSIONS);ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1]*1.10 if key.endswith('db') else 1.10)
    fig.suptitle('20 saved project PNGs: source consistency on common original-input overlap', fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.text(.5,.02,'Higher = closer to original pixels, NOT ground-truth geometry or aesthetic quality. E1/E2 use identical evaluation regions.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.96])
    fig.savefig(OUT/'project_scores.png',dpi=180)
    fig.savefig(OUT/'project_scores.pdf')
    plt.close(fig)
    report = OUT/'report.html'
    text = report.read_text()
    marker = '<h2>20개 최종 결과의 원본 일치도</h2>'
    plot = '<p><a href="project_scores.png">발표용 그래프 PNG</a> · <a href="project_scores.pdf">그래프 PDF</a></p><img src="project_scores.png" alt="최종 PNG 20개의 원본 일치도 점수 비교">'
    if 'src="project_scores.png"' not in text:
        report.write_text(text.replace(marker,marker+plot),encoding='utf-8')
    print('Saved',OUT/'project_scores.png')


if __name__=='__main__':
    main()
