#!/usr/bin/env python3
"""解析 .vsqx 文件并合成音频"""

import sys, json, os, math
import xml.etree.ElementTree as ET
import numpy as np
import torch
import soundfile as sf
import yaml

# ── 配置 ──
SR = 24000
HOP = 256
FR = SR / HOP  # 93.75 帧/秒
TPB = 480      # VSQ4 标准 tick per beat

# ── X-SAMPA → OpenCPOP 音素映射 ──
# 基于 OpenCPOP 60 音素集
X2OC = {
    # 声母 (consonants)
    'p': 'b', 'p_h': 'p', 'b': 'b', 't': 'd', 't_h': 't', 'd': 'd',
    'k': 'g', 'k_h': 'k', 'g': 'g',
    'ts': 'z', 'ts_h': 'c', 'dz': 'z', 
    'ts`': 'zh', 'ts`_h': 'ch', 'dz`': 'zh',
    'ts\\': 'j', 'ts\\_h': 'q', 'dz\\': 'j',
    'f': 'f', 's': 's', 'S': 'sh', 'x': 'x', 'h': 'h',
    'm': 'm', 'n': 'n', 'l': 'l', 'r': 'r',
    # 韵母 (vowels/syllabics)
    'a': 'a', 'A': 'a',
    'o': 'o', 'O': 'o',
    '7': 'e', '@': 'e', 'e': 'e', 'E': 'e',
    'i': 'i', 'I': 'i',
    'u': 'u', 'U': 'u',
    'y': 'v',  # ü in X-SAMPA is 'y'
    'aU': 'ao', 'a_I': 'ai', 'aI': 'ai',
    'ei': 'ei', 'eI': 'ei',
    'ou': 'ou', '@U': 'ou',
    'ia': 'ia', 'iA': 'ia',
    'ie': 'ie', 'iE': 'ie',
    'ua': 'ua', 'uA': 'ua',
    'uo': 'uo', 'uO': 'uo',
    've': 've', 'vE': 've',
    'iau': 'iao', 'iAU': 'iao',
    'iou': 'iu', 'i@U': 'iu',
    'uai': 'uai',
    'uei': 'ui', 'u@I': 'ui',
    'an': 'an', 'a_n': 'an',
    'en': 'en', '@_n': 'en',
    'ian': 'ian', 'iE_n': 'ian',
    'in': 'in', 'i_n': 'in',
    'uan': 'uan', 'uA_n': 'uan',
    'yn': 'vn', 'y_n': 'vn',  # ün
    'ang': 'ang', 'A_N': 'ang',
    'eng': 'eng', '@_N': 'eng',
    'iang': 'iang', 'iA_N': 'iang',
    'ing': 'ing', 'i_N': 'ing', 'iN': 'ing',
    'ong': 'ong', 'U_N': 'ong',
    'iong': 'iong', 'yU_N': 'iong',
    'er': 'er', '@r': 'er',
}

# 元音集 (用于辅音/元音时长分配)
VOWELS = set(X2OC.values()) - {'b','p','m','f','d','t','n','l','g','k','h',
                                 'j','q','x','zh','ch','sh','r','z','c','s',
                                 'y','w','SP','AP'}

def parse_vsqx(path):
    """解析 VSQX 文件，返回音符列表和 tempo(BPM)"""
    tree = ET.parse(path)
    root = tree.getroot()
    
    # 处理默认命名空间
    ns_uri = ''
    if '}' in root.tag:
        ns_uri = root.tag.split('}')[0].strip('{')
        ET.register_namespace('', ns_uri)
    
    # 找 tempo  
    tempo_el = None
    for el in root.iter():
        if el.tag.endswith('tempo') or (ns_uri and el.tag == f'{{{ns_uri}}}tempo'):
            tempo_el = el
            break
    
    if tempo_el is not None:
        v_el = None
        for el in tempo_el.iter():
            if el.tag.endswith('v') or (ns_uri and el.tag == f'{{{ns_uri}}}v'):
                v_el = el
                break
        tempo_raw = int(v_el.text) if v_el is not None else 6500
    else:
        tempo_raw = 6500
    bpm = tempo_raw / 100.0
    
    # 找到所有 note
    notes = []
    for el in root.iter():
        tag = el.tag
        # 去掉命名空间
        if '}' in tag:
            local = tag.split('}')[1]
        else:
            local = tag
        if local != 'note':
            continue
        
        def find_child(parent, child_name):
            for c in parent:
                ctag = c.tag
                if '}' in ctag:
                    clocal = ctag.split('}')[1]
                else:
                    clocal = ctag
                if clocal == child_name:
                    return c
            return None
        
        t_el = find_child(el, 't')
        dur_el = find_child(el, 'dur')
        n_el = find_child(el, 'n')
        y_el = find_child(el, 'y')
        p_el = find_child(el, 'p')
        
        t = int(t_el.text) if t_el is not None else 0
        dur = int(dur_el.text) if dur_el is not None else 0
        midi_n = int(n_el.text) if n_el is not None else 60
        lyric = y_el.text if y_el is not None else ''
        xsampa = p_el.text if p_el is not None else ''
        
        notes.append({
            'tick': t, 'dur_ticks': dur, 'midi': midi_n,
            'lyric': lyric, 'xsampa': xsampa
        })
    
    return notes, bpm

def xsampa_to_phonemes(xsampa):
    """X-SAMPA → OpenCPOP 音素列表"""
    if xsampa in ('-', '?', ''):
        return ['SP']
    
    # 尝试整个串匹配（复合韵母）
    if xsampa in X2OC:
        return [X2OC[xsampa]]
    
    # 按空格分割（多音素情况如 "k u", "ts` i`"）
    parts = xsampa.strip().split()
    result = []
    for part in parts:
        if not part:
            continue
        # 尝试直接匹配
        if part in X2OC:
            result.append(X2OC[part])
        elif part.lower() in X2OC:
            result.append(X2OC[part.lower()])
        else:
            # 尝试最长前缀匹配（如 ts\_h → q）
            best = None
            for k in sorted(X2OC.keys(), key=len, reverse=True):
                if part.startswith(k) or part.lower().startswith(k.lower()):
                    best = X2OC[k]
                    break
            if best:
                result.append(best)
            else:
                print(f"  [警告] 无法映射音素: '{part}' (来自 '{xsampa}')，使用 'SP'")
                result.append('SP')
    
    return result if result else ['SP']

def create_synthesis(notes, bpm, device='cuda'):
    """从音符列表构建合成输入"""
    tick_sec = 60.0 / (bpm * TPB)  # 每 tick 的秒数
    total_frames = 0
    segments = []
    
    for i, note in enumerate(notes):
        start_sec = note['tick'] * tick_sec
        dur_sec = note['dur_ticks'] * tick_sec
        f0 = 440.0 * (2 ** ((note['midi'] - 69) / 12))
        
        # 计算段间间隔
        if i > 0:
            gap = start_sec - (notes[i-1]['tick'] + notes[i-1]['dur_ticks']) * tick_sec
            if gap > 0.02:
                segments.append(('rest', gap, 0.0, 'SP', ['SP']))
        
        ph = xsampa_to_phonemes(note['xsampa'])
        segments.append(('note', dur_sec, f0, note['lyric'], ph))
    
    # 累计到下一个 note 起始的间隔
    # 已经在上面的循环中处理了
    
    # 生成长音素序列和音符帧
    all_phonemes = []
    seg_ph_counts = []
    seg_frames = []
    
    for seg_type, seg_dur, f0v, lyric, phs in segments:
        n_frames = max(1, int(seg_dur * FR + 0.5))
        all_phonemes.extend(phs)
        seg_ph_counts.append(len(phs))
        seg_frames.append(n_frames)
    
    # 手动时长分配 (辅音~9%, 元音~91%)
    C_W = 0.09; V_W = 0.91
    ph_dur_manual = []
    for s_idx, n_ph in enumerate(seg_ph_counts):
        nf = seg_frames[s_idx]
        if n_ph <= 1:
            ph_dur_manual.append(nf)
            continue
        ph_start = sum(seg_ph_counts[:s_idx])
        ph_list = all_phonemes[ph_start:ph_start+n_ph]
        n_cons = sum(1 for p in ph_list if p not in VOWELS and p != 'SP')
        n_vows = n_ph - n_cons
        if n_cons == 0 or n_vows == 0:
            base = nf // n_ph; rem = nf % n_ph
            for j in range(n_ph):
                ph_dur_manual.append(base + (1 if j < rem else 0))
        else:
            ctw = n_cons * C_W; vtw = n_vows * V_W; tw = ctw + vtw
            cf = max(1, int(nf * ctw / tw)); vf = nf - cf
            cp = cf // n_cons; cr = cf % n_cons
            cd = [cp + (1 if i < cr else 0) for i in range(n_cons)]
            vp = vf // n_vows; vr = vf % n_vows
            vd = [vp + (1 if i < vr else 0) for i in range(n_vows)]
            ci = 0; vi = 0
            for p in ph_list:
                if p not in VOWELS and p != 'SP':
                    ph_dur_manual.append(cd[ci]); ci += 1
                else:
                    ph_dur_manual.append(vd[vi]); vi += 1
    
    total_pf = sum(ph_dur_manual)
    
    # F0 曲线
    f0_frames = []
    for s_idx, (seg_type, seg_dur, f0v, lyric, phs) in enumerate(segments):
        nf = seg_frames[s_idx]
        f0_frames.extend([f0v] * nf)
    f0_frames = np.array(f0_frames[:total_pf], dtype=np.float32)
    
    # 音符帧
    note_frames = [seg_frames[i] for i, (st, _, _, _, _) in enumerate(segments) if st == 'note']
    
    return all_phonemes, ph_dur_manual, f0_frames, note_frames


def main():
    # 加载 phoneme_map
    pm_path = '/root/autodl-tmp/yuaz_sgr/data/preprocessed/phoneme_map.json'
    with open(pm_path) as f:
        pm = json.load(f)
    p2i = pm['p2i']
    
    # 解析 VSQX
    vsqx_path = '/root/autodl-tmp/虚拟.vsqx'
    print(f"解析 {vsqx_path}...")
    notes, bpm = parse_vsqx(vsqx_path)
    print(f"  BPM: {bpm}, 音符数: {len(notes)}")
    print(f"  歌词: {' '.join(n['lyric'] for n in notes)}")
    
    # 创建合成数据
    all_phonemes, ph_dur_manual, f0_frames, note_frames = create_synthesis(notes, bpm)
    
    # 映射音素 ID
    ph_ids = []
    for p in all_phonemes:
        pid = p2i.get(p, 0)
        if pid == 0:
            print(f"  [警告] 未找到音素 '{p}'，使用 SP(0)")
        ph_ids.append(pid)
    
    total_frames = sum(ph_dur_manual)
    print(f"\n  音素数: {len(ph_ids)}, 总帧数: {total_frames}, 音符数: {len(note_frames)}")
    print(f"  F0范围: {f0_frames.min():.0f}-{f0_frames.max():.0f} Hz")
    print(f"  预测时长: {total_frames/FR:.1f}s")
    
    # 检查 f0 帧数是否匹配
    if len(f0_frames) > total_frames:
        f0_frames = f0_frames[:total_frames]
    elif len(f0_frames) < total_frames:
        f0_frames = np.pad(f0_frames, (0, total_frames - len(f0_frames)), 'edge')
    
    # 加载模型
    sys.path.insert(0, '/root/autodl-tmp/yuaz_sgr')
    from models.svs_model import SVSModel
    
    config = {}
    for name in ['base', 'model']:
        with open(f'/root/autodl-tmp/yuaz_sgr/config/{name}.yaml') as f:
            config.update(yaml.safe_load(f))
    config['conditioning']['num_phonemes'] = pm['num_phonemes']
    
    ckpt_files = sorted([f for f in os.listdir('/root/autodl-tmp/yuaz_sgr/checkpoints')
                         if f.startswith('model_') and f.endswith('.pt')])
    if not ckpt_files:
        print("没有检查点文件!")
        return
    
    model = SVSModel(config).to(device := 'cuda')
    model.eval()
    ckpt_path = f'/root/autodl-tmp/yuaz_sgr/checkpoints/{ckpt_files[-1]}'
    ckpt = torch.load(ckpt_path, map_location=device)
    msd = model.state_dict()
    for k, v in ckpt['model'].items():
        ck = k.replace('_orig_mod.', '')
        if ck in msd and v.shape == msd[ck].shape:
            msd[ck] = v
    model.load_state_dict(msd, strict=False)
    print(f"\n加载模型: {ckpt_files[-1]} (step {ckpt.get('step', 0)})")
    
    # ── 推理 ──
    phoneme_t = torch.LongTensor(ph_ids).unsqueeze(0).cuda()
    dur_t = torch.LongTensor(ph_dur_manual).unsqueeze(0).cuda()
    f0_t = torch.FloatTensor(f0_frames).unsqueeze(0).unsqueeze(0).cuda()
    nl_t = torch.LongTensor(note_frames).unsqueeze(0).cuda()
    
    OUT = '/root/autodl-tmp/vsqx_test'
    os.makedirs(OUT, exist_ok=True)
    
    # Mode 1: 原版 F0
    print(f"\n[1/3] 原版 F0...")
    with torch.no_grad():
        w1, _, _, _, _, _ = model.forward_synthesis(
            phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
            use_predicted_f0=False, predict_durations=False
        )
    w1 = w1.squeeze().cpu().numpy()
    w1 = w1 / (np.abs(w1).max() + 1e-8)
    sf.write(f'{OUT}/orig_f0.wav', w1, SR)
    print(f"  Done: {len(w1)/SR:.1f}s")
    
    # Mode 2: 预测 F0
    print(f"[2/3] 预测 F0...")
    with torch.no_grad():
        w2, _, _, _, _, _ = model.forward_synthesis(
            phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
            use_predicted_f0=True, predict_durations=False
        )
    w2 = w2.squeeze().cpu().numpy()
    w2 = w2 / (np.abs(w2).max() + 1e-8)
    sf.write(f'{OUT}/pred_f0.wav', w2, SR)
    print(f"  Done: {len(w2)/SR:.1f}s")
    
    # Mode 3: 预测音素时长
    print(f"[3/3] 预测音素时长...")
    with torch.no_grad():
        w3, _, _, _, _, _ = model.forward_synthesis(
            phoneme_t, dur_t, f0_t, n_steps=1, note_lengths=nl_t,
            use_predicted_f0=False, predict_durations=True
        )
    w3 = w3.squeeze().cpu().numpy()
    w3 = w3 / (np.abs(w3).max() + 1e-8)
    sf.write(f'{OUT}/pred_dur.wav', w3, SR)
    print(f"  Done: {len(w3)/SR:.1f}s")
    
    print(f"\n完成: {OUT}/")
    for f in ['orig_f0.wav', 'pred_f0.wav', 'pred_dur.wav']:
        fpath = os.path.join(OUT, f)
        if os.path.exists(fpath):
            print(f"  {fpath}")


if __name__ == '__main__':
    main()
