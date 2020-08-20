import plotly.graph_objects as go

from plotly.subplots import make_subplots
import numpy as np
import minerl
import collections
import plotly.io as pio
pio.renderers.default = "browser"

def main(dataname):
    data = minerl.data.make(
    dataname,
    data_dir='.')
    times = []

            if done:
                times.append(meta['duration_ms'])

    fig = go.Figure()
    fig.add_trace(go.Box(y=times))
    fig.show()
    print(times)
    bins = np.linspace(np.min(times), np.max(times), num=64)
    bots = np.percentile(times, 25)
    binsize = bins[1] - bins[0]
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=times, name='all',opacity=0.75,
    xbins=dict(
        start=np.min(times),
        end=np.max(times),
        size=binsize
    )))
    
    index = np.array(times) <= bots
    bot_times = np.array(times)
    bot_times = np.array(times)[index]
    fig1.add_trace(go.Histogram(x=bot_times, name='bot',opacity=0.75,
    xbins=dict(
        start=np.min(times),
        end=np.max(times),
        size=binsize
    )))
    
    index = np.array(times) > bots
    top_times = np.array(times)
    top_times = np.array(times)[index]
    fig1.add_trace(go.Histogram(x=top_times, name='top',opacity=0.75,
    xbins=dict(
        start=np.min(times),
        end=np.max(times),
        size=binsize
    )))
    fig1.update_layout(barmode='overlay', title_text='disturbution of run times')
    
    fig1.show()
    return bots
    
if __name__ == '__main__':
    main('MineRLObtainDiamond-v0')