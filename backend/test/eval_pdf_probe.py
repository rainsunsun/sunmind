# -*- coding: utf-8 -*-
"""探测 PDF：页数 / 文字量 / 提取质量（结果写文件避免控制台乱码）"""
from pypdf import PdfReader

PDF = r"C:\Users\rainsunsun\Desktop\概率论与数理统计教程+第3版+习题与解答+(茆诗松).pdf"
OUT = r"C:\Users\rainsunsun\Desktop\learn\echomind\EchoMind\backend\test\pdf_preview.txt"

doc = PdfReader(PDF)
n = len(doc.pages)
total = 0
per_page = []
samples = []
for i in range(n):
    t = doc.pages[i].extract_text() or ""
    total += len(t)
    per_page.append(len(t))
    if i < 6:
        samples.append((i + 1, t))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"文件: {PDF}\n")
    f.write(f"页数: {n}\n")
    f.write(f"总字符数: {total}\n")
    f.write(f"平均每页字符: {total // max(n, 1)}\n")
    f.write("=" * 60 + "\n")
    for pg, t in samples:
        f.write(f"\n----- 第 {pg} 页（前 500 字）-----\n")
        f.write(t[:500])
        f.write("\n")

print("DONE pages=", n, "chars=", total)
