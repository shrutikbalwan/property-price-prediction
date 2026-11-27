"""Portable SVG charts for the notebooks and reports (no plotting dependency).

Charts contain their own titles, axis labels, ticks, and accessible descriptions.
SVG is used so the generated figures remain sharp when exported or printed.
"""
from html import escape
from pathlib import Path
import numpy as np

WIDTH, HEIGHT = 900, 540


def text(x,y,value,size=13,anchor='start',color='#34445a'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>'


def save(path,title,body,description=''):
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
         f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img"><title>{escape(title)}</title>'
         f'<desc>{escape(description)}</desc><rect width="100%" height="100%" fill="#fff"/>'
         '<g font-family="Arial, sans-serif">'+text(35,37,title,22)+body+'</g></svg>')
    Path(path).write_text(svg,encoding='utf-8')
    return svg


def bars(path,title,labels,values,xlabel='Value'):
    values=np.asarray(values,dtype=float)
    n=len(labels)
    height=min(42,360/max(n,1))
    scale=max(float(np.max(np.abs(values))),1e-12)
    body=''
    for i,(label,value) in enumerate(zip(labels,values)):
        y=78+i*height
        body+=text(270,y+height*0.6,str(label)[:35],12,'end')
        width=abs(value)/scale*450
        body+=f'<rect x="285" y="{y}" width="{width}" height="{height*.72}" fill="{"#267c95" if value>=0 else "#b9504e"}"/>'
        body+=text(292+width,y+height*.55,f'{value:,.4g}',12)
    body+=text(285,490,xlabel,14)
    return save(path,title,body,'Horizontal bar chart. Values are printed beside each bar. Negative values are red and carry a minus sign.')


def scatter(path,title,x,y,xlabel,ylabel,diagonal=False,connect=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    finite=np.isfinite(x)&np.isfinite(y)
    x,y=x[finite],y[finite]
    if len(x)>1800 and not connect:
        chosen=np.random.default_rng(42).choice(len(x),1800,replace=False)
        x,y=x[chosen],y[chosen]
    xmin,xmax=float(x.min()),float(x.max())
    ymin,ymax=float(y.min()),float(y.max())
    if diagonal:
        xmin=ymin=min(xmin,ymin); xmax=ymax=max(xmax,ymax)
    def sx(v): return 100+(v-xmin)/max(xmax-xmin,1e-12)*740
    def sy(v): return 440-(v-ymin)/max(ymax-ymin,1e-12)*350
    body=''
    for a in np.linspace(0,1,6):
        xx=100+a*740; yy=440-a*350
        body+=f'<path d="M {xx} 90 V 440 M 100 {yy} H 840" stroke="#e2e8ef"/>'
        body+=text(xx,463,f'{xmin+a*(xmax-xmin):,.3g}',12,'middle')
        body+=text(88,yy+4,f'{ymin+a*(ymax-ymin):,.3g}',12,'end')
    if diagonal:
        body+='<path d="M 100 440 L 840 90" stroke="#c65b45" stroke-width="2"/>'
    if connect:
        points=' '.join(f'{sx(a):.2f},{sy(b):.2f}' for a,b in zip(x,y))
        body+=f'<polyline points="{points}" fill="none" stroke="#267c95" stroke-width="2"/>'
    for a,b in zip(x,y):
        body+=f'<circle cx="{sx(a):.2f}" cy="{sy(b):.2f}" r="2" fill="#267c95" opacity=".35"/>'
    body+=text(470,502,xlabel,14,'middle')+text(100,72,ylabel,14)
    return save(path,title,body,'Scatter plot; at most 1,800 deterministically sampled points are shown.')


def histogram(path,title,values,xlabel,bins=18):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    counts,edges=np.histogram(values,bins=bins)
    body=''; maximum=max(counts.max(),1)
    for i,count in enumerate(counts):
        x=100+i*740/bins; h=count/maximum*340
        body+=f'<rect x="{x}" y="{440-h}" width="{740/bins-2}" height="{h}" fill="#267c95"/>'
    for a in np.linspace(0,1,6):
        body+=text(100+a*740,465,f'{edges[0]+a*(edges[-1]-edges[0]):,.3g}',12,'middle')
        body+=text(88,444-a*340,f'{a*maximum:,.0f}',12,'end')
    body+=text(470,505,xlabel,14,'middle')+text(100,75,'Count',14)
    return save(path,title,body,f'Histogram of {len(values)} nonmissing observations, {bins} bins.')
