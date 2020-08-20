import plotly.graph_objects as go
import numpy as np
import minerl
import collections
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
pio.renderers.default = "browser"

def main():
    
    #note need to run getminerldata in the same dict first
    data = minerl.data.make(
    "MineRLObtainIronPickaxe-v0",
    data_dir='.')
    times = []
    for x in data.get_trajectory_names():
        for current_state, action, reward, next_state, done, meta in data.load_data(x,include_metadata=True):
            if done:
                times.append(meta['duration_ms'])
    bots = np.percentile(times, 20)                
    print("1")
    data = minerl.data.make(
        'MineRLObtainIronPickaxe-v0',
        data_dir='.')
    print(data)
    possilbe_actions = ["attack","back","forward","jump","left","right","sneak","sprint"]
    print(possilbe_actions)
    action_map = {"attack":0,"back":1,"forward":2,"jump":3,"left":4,"right":5,"sneak":6,"sprint":7}
    print(action_map)

    d = collections.defaultdict(list)
    r = collections.defaultdict(list)
    reward_list = []
    norm = []
    
    
    d_bot = collections.defaultdict(list)
    r_bot = collections.defaultdict(list)
    reward_list_bot = []
    norm_bot = []
    
    d_rest = collections.defaultdict(list)
    r_rest = collections.defaultdict(list)
    reward_list_rest = []
    norm_rest = []
    
    camera_list = []
    issprint_list = []
    issattck_list = []

    counter =  0
    r_lenght = 0
    print(counter)
    for x in data.get_trajectory_names():
        for current_state, action, reward, next_state, done, meta in data.load_data(x,include_metadata=True):

                for x in possilbe_actions:
                    
                    if counter < len(d[x]):
                        d[x][counter] += action[x]
                        r[x][counter] += reward*action[x]
                        
                        if meta['duration_ms'] <= bots:
                            d_bot[x][counter] += action[x]
                            r_bot[x][counter] += reward*action[x]
                        else:
                            d_rest[x][counter] += action[x]
                            r_rest[x][counter] += reward*action[x]
                    else:
                        d[x].append(action[x])
                        r[x].append(reward*action[x])
                        
                        if meta['duration_ms'] <= bots:
                            d_bot[x].append(action[x])
                            r_bot[x].append(reward*action[x])
                            d_rest[x].append(0)
                            r_rest[x].append(0)
                        else:
                            d_rest[x].append(action[x])
                            r_rest[x].append(reward*action[x])
                            d_bot[x].append(0)
                            r_bot[x].append(0)

                if counter < len(reward_list):
                    reward_list[counter] += reward
                    if meta['duration_ms'] <= bots:
                        reward_list_bot[counter] += reward
                    else:
                        reward_list_rest[counter] += reward
                else:
                    reward_list.append(reward)
                    if meta['duration_ms'] <= bots:
                        reward_list_bot.append(reward)
                        reward_list_rest.append(0)
                    else:
                        reward_list_rest.append(reward)
                        reward_list_bot.append(0)
        
                if counter < len(norm):
                    norm[counter] += 1
                    if meta['duration_ms'] <= bots:
                        norm_bot[counter] += 1
                    else:
                        norm_rest[counter] += 1
                else:
                    norm.append(1)
                    if meta['duration_ms'] <= bots:
                        norm_bot.append(1)
                        norm_rest.append(0)
                    else:
                        norm_rest.append(1)
                        norm_bot.append(0)                       
                camera_list.append(action['camera'])
                issprint_list.append(action['sprint'])
                issattck_list.append(action['attack'])
                if done:
                    counter = 0
                    camera_list.append([meta['duration_ms'] <= bots])
                    issattck_list.append([meta['duration_ms'] <= bots])
                    issprint_list.append(meta['duration_ms'])
                counter += 1
                
            
                
    times = list(times)     
    fig = go.Figure()
    fig.add_trace(go.Box(y=times))
    fig.update_layout(autosize=False,
              width=500, height=500,
              margin=dict(l=65, r=50, b=65, t=90))
    fig.update_layout(
        title_text='Disturbution of run times',
        xaxis_title=" ",
        yaxis_title="Time (milliseconds)")
    fig.show()
    print(times)
    bins = np.linspace(np.min(times), np.max(times), num=64)

    binsize = bins[1] - bins[0]
    fig1 = go.Figure()
    # fig1.add_trace(go.Histogram(x=times, name='all',opacity=0.75,
    # xbins=dict(
    #     start=np.min(times),
    #     end=np.max(times),
    #     size=binsize
    # )))
    
    index = np.array(times) <= bots
    bot_times = np.array(times)
    bot_times = np.array(times)[index]
    fig1.add_trace(go.Histogram(x=bot_times, name='Fastest',opacity=0.75,
    xbins=dict(
        start=np.min(times),
        end=np.max(times),
        size=binsize
    )))
    
    index = np.array(times) > bots
    top_times = np.array(times)
    top_times = np.array(times)[index]
    fig1.add_trace(go.Histogram(x=top_times, name='Rest',opacity=0.75,
    xbins=dict(
        start=np.min(times),
        end=np.max(times),
        size=binsize
    )))
    fig1.update_layout(barmode='stack')
    fig1.update_layout(
        title_text='Disturbution of run times',
        yaxis_title="Game play samples",
        xaxis_title="Time (milliseconds)")
    fig1.show()

    fig_ = go.Figure()
    print(times)
    index = np.array(times) <= bots
    bot_times = np.array(times)
    bot_times = np.array(times)[index]
    bins = np.linspace(np.min(bot_times), np.max(bot_times), num=8)

    binsize = bins[1] - bins[0]

    fig_.add_trace(go.Histogram(x=bot_times, name='bot',opacity=0.75,
    xbins=dict(
        start=np.min(bot_times),
        end=np.max(bot_times),
        size=binsize
    )))
    
    fig_.update_layout(barmode='stack', title_text='Disturbution of run times')
    
    fig_.show()                    
                
                
                
    
    counts = np.zeros((len(d['back']),len(possilbe_actions)))   
    for x in possilbe_actions:     
        counts[:,action_map[x]] = np.array(d_bot[x])/np.linalg.norm(np.array(d_bot[x]))
    fig2 = make_subplots(rows=1, cols=2,specs=[[{"type": "scene"}, {"type": "scene"}]])
    fig2.add_trace(go.Surface(z=counts,x=possilbe_actions),row=1,col=1)
    
    counts = np.zeros((len(d['back']),len(possilbe_actions)))   
    for x in possilbe_actions:     
        counts[:,action_map[x]] = np.array(d_rest[x])/np.linalg.norm(np.array(d_rest[x]))
    fig2.add_trace(go.Surface(z=counts,x=possilbe_actions,showscale=False),row=1,col=2)
    fig2.update_layout(title='Action over time landscape')
    fig2.update_layout(scene = dict(
                    xaxis_title='Action',
                    yaxis_title='Time (milliseconds)',
                    zaxis_title='Action counts'))
    fig2.show()
    
    # counts_r = np.zeros((len(r['back']),len(possilbe_actions)))   
    # for x in possilbe_actions:     
    #     counts_r[:,action_map[x]] = np.array(r[x])
    # fig3 = go.Figure(data=[go.Surface(z=counts_r)])
    # fig3.update_layout(title='Reward-Action landscape', autosize=False,
    #                   width=500, height=500,
    #                   margin=dict(l=65, r=50, b=65, t=90))
    # fig3.show()
    fig4 = go.Figure()
    total_actions = np.array([0]*len(possilbe_actions))
    total_actions_top = np.array([0]*len(possilbe_actions))
    total_actions_rest = np.array([0]*len(possilbe_actions))
    for k,x in enumerate(possilbe_actions):
        total_actions[k] = sum(d[x])
        total_actions_top[k] = sum(d_bot[x])
        total_actions_rest[k] = sum(d_rest[x])
    total_actions = total_actions/sum(total_actions)
    total_actions_top = total_actions_top/sum(total_actions_top)
    total_actions_rest = total_actions_rest/sum(total_actions_rest)
    
    fig4.add_trace(go.Bar(x=possilbe_actions,y=total_actions_top, name="Actions taken in fastest run"))
    fig4.add_trace(go.Bar(x=possilbe_actions,y=total_actions_rest, name="Actions taken in rest runs"))
    fig4.update_layout(
        title_text='Ratios of actions taken',
        xaxis_title="Actions",
        yaxis_title="Ratio of action")
    fig4.show()
    
    # fig5 = go.Figure()
    # y_range_norm = [x for x in range(1,len(norm))]
    # y_range_bot_norm = [x for x in range(1,len(norm_bot))]
    # y_range_rest_norm = [x for x in range(1,len(norm_rest))]
    # fig5.add_trace(go.Scatter(y=norm,x=y_range_norm, name="average reward at each step",opacity=0.5))
    # fig5.add_trace(go.Scatter(y=norm_bot,x=y_range_bot_norm, name="average reward at each step",opacity=0.5))
    # fig5.add_trace(go.Scatter(y=norm_rest,x=y_range_rest_norm, name="average reward at each step",opacity=0.5))
    # fig5.show()
    
    last_index = 0
    bot_MSD = []
    rest_MSD = []
    rest_auto = []
    bot_auto = []
    bot_sprinter = []
    rest_sprinter = []
    duration = []
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[result.size//2:]
    
    def xcorr(x):
          """FFT based autocorrelation function, which is faster than numpy.correlate"""
          # x is supposed to be an array of sequences, of shape (totalelements, length)
          fftx = np.fft.fft(x, n=(x.shape[1]*2-1), axis=1)
          ret = np.fft.ifft(fftx * np.conjugate(fftx), axis=1)
          ret = np.fft.fftshift(ret, axes=1)
          return ret
    fig6 = make_subplots(rows=1, cols=2,subplot_titles=("Fastest", "Rest"))
    #fig6.add_trace(go.Scatter(y=[-100,-100,100,100],x=[-100,100,100,-100],opacity=0),col=1,row=1)
    #fig6.add_trace(go.Scatter(y=[-100,-100,100,100],x=[-100,100,100,-100],opacity=0),col=2,row=1)
    bot_dur = []
    rest_dur = []
    rest_attack = []
    bot_attack = []
    for k,x in enumerate(camera_list):
        if len(x) < 2:
            trace = camera_list[last_index+1:k]
            sprint_ratio = sum(issprint_list[last_index+1:k])/len(trace)
            attack_ratio = sum(issattck_list[last_index+1:k])/len(trace)
            last_index = k
            duration.append(issprint_list[k])
            if x[0]:
                bot_dur.append(issprint_list[k])
                bot_sprinter.append(sprint_ratio)
                bot_attack.append(attack_ratio)
                xdata = np.array([a[0] for a in trace])
                ydata = np.array([a[1] for a in trace])
                bot_auto.append(np.mean(autocorr(np.sqrt(xdata**2 + ydata**2))))
                #bot_auto.append(np.mean(xcorr(np.asarray((trace)))))
                fig6.add_trace(go.Scatter(y=ydata,x=xdata,opacity=0.08),row=1,col=1)
                x = np.sqrt(xdata**2 + ydata**2)
                #x = x[x>0.01]
                bot_MSD.append(np.mean(x))
            else:
                rest_sprinter.append(sprint_ratio)
                rest_attack.append(attack_ratio)
                rest_dur.append(issprint_list[k])
                xdata = np.array([a[0] for a in trace])
                ydata = np.array([a[1] for a in trace])
                rest_auto.append(np.mean(autocorr(np.sqrt(xdata**2 + ydata**2))))
                #rest_auto.append(np.mean(xcorr(np.asarray((trace)))))
                fig6.add_trace(go.Scatter(y=ydata,x=xdata,opacity=0.08),row=1,col=2)
                x = np.sqrt(xdata**2 + ydata**2)
                #x = x[x>0.01]
                rest_MSD.append(np.mean(x))
                
    fig6.update_layout(showlegend=False)
    fig6.update_layout(title_text='Camera movments')
    fig6.update_xaxes(title_text="X displacement",range=[100, -100], row=1, col=1)
    fig6.update_xaxes(title_text="X displacement",range=[100, -100], row=1, col=2)
    fig6.update_yaxes(title_text="Y displacement",range=[100, -100], row=1, col=1)
    fig6.update_yaxes(title_text="Y displacement",range=[100, -100], row=1, col=2)
    rest_sprinter = np.array(rest_sprinter)/np.linalg.norm(np.array(rest_sprinter))
    bot_sprinter = np.array(bot_sprinter)/np.linalg.norm(np.array(bot_sprinter))
    rest_attack = np.array(rest_attack)/np.linalg.norm(np.array(rest_attack))
    bot_attack = np.array(bot_attack)/np.linalg.norm(np.array(bot_attack))
    rest_auto = np.array(rest_auto)/np.linalg.norm(np.array(rest_auto))
    bot_auto = np.array(bot_auto)/np.linalg.norm(np.array(bot_auto))
    bot_MSD = np.array(bot_MSD)/np.linalg.norm(np.array(bot_MSD))
    rest_MSD = np.array(rest_MSD)/np.linalg.norm(np.array(rest_MSD))
    fig6.show()
    MSD_max = max(np.max(rest_MSD),np.max(bot_MSD))
    MSD_min = min(np.min(rest_MSD),np.min(bot_MSD))
    bins = np.linspace(MSD_min, MSD_max, num=64)

    binsize = bins[1] - bins[0]
    fig8 = go.Figure()
    fig8.add_trace(go.Histogram(x=rest_MSD, name='Rest',opacity=0.75,
    xbins=dict(
        start=MSD_min,
        end=MSD_max,
        size=binsize
    )))
    
    fig8.add_trace(go.Histogram(x=bot_MSD, name='Fastest',opacity=0.75,
    xbins=dict(
        start=MSD_min,
        end=MSD_max,
        size=binsize
    )))
    fig8.update_layout(barmode='overlay')
    fig8.update_layout(
        title_text='Disturbution of mean camera movments',
        xaxis_title="Mean camera movment",
        yaxis_title="Game play samples")
    fig8.show()
    
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=bot_sprinter,y=bot_attack, mode='markers',name = 'Fastest'))
    fig9.add_trace(go.Scatter(x=rest_sprinter,y=rest_attack, mode='markers',name = 'Rest'))
    fig9.update_layout(
        title_text='Projected expert non-expert split',
        xaxis_title="Sprinting ratio",
        yaxis_title="Attacking ratio")
    fig9.show()
    
    fig9_1 = go.Figure()
    fig9_1.add_trace(go.Scatter3d(x=bot_sprinter,y=bot_attack,z=bot_MSD, mode='markers',name = 'Fastest'))
    fig9_1.add_trace(go.Scatter3d(x=rest_sprinter,y=rest_attack,z=rest_MSD, mode='markers',name = 'Rest'))
    fig9_1.update_layout(title_text='Projected expert non-expert split', scene = dict(
            xaxis_title='Sprinting ratio',
            yaxis_title='Attacking ratio',
            zaxis_title='Median crema movment'))
    fig9_1.show()
    
    x_data = np.asarray(list(zip(bot_sprinter,bot_attack)) +list(zip(rest_sprinter,rest_attack)))
    x_data_1 = np.asarray(list(zip(bot_sprinter,bot_attack,bot_MSD)) +list(zip(rest_sprinter,rest_attack,rest_MSD)))
    y_data = np.array([1]*len(bot_MSD) + [0]*len(rest_MSD))
    
    # logreg = LogisticRegression(penalty='elasticnet',C=0.00001,solver='saga',l1_ratio=0.5,max_iter=200)
    # logreg.fit(x_data, y_data)
    # print(logreg)
    # print(logreg.coef_)
    # print(logreg.intercept_)
    #X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    machine = SVC(C=0.5).fit(x_data, y_data)
    machine_1 = SVC(C=0.5).fit(x_data_1, y_data)
    # print(SVC.score(X_test,y_test))
    Z = machine.predict(x_data)
    new_bot = []
    new_bot_pt = []
    new_rest = []
    new_rest_pt = []
    for k,z in enumerate(Z):
        if z > 0:
            new_bot.append(duration[k])
            new_bot_pt.append(x_data[k])
        else:
            new_rest.append(duration[k])
            new_rest_pt.append(x_data[k])
            
    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(y=[x[1] for x in new_bot_pt],x=[x[0] for x in new_bot_pt], mode='markers',name = 'Fastest'))
    fig10.add_trace(go.Scatter(y=[x[1] for x in new_rest_pt],x=[x[0] for x in new_rest_pt], mode='markers',name = 'Rest'))
    fig10.update_layout(
        title_text='Modelled projected expert non-expert split',
        xaxis_title="Sprinting ratio",
        yaxis_title="Attacking ratio")
    fig10.show()
            
    dur_max = max(duration)
    dur_min = min(duration)
    bins = np.linspace(dur_min, dur_max, num=64)

    binsize = bins[1] - bins[0]
    fig11 = go.Figure()
    fig11.add_trace(go.Histogram(x=new_bot, name='1',opacity=0.5,
    xbins=dict(
        start=dur_min,
        end=dur_max,
        size=binsize
    )))
    
    fig11.add_trace(go.Histogram(x=new_rest, name='0',opacity=0.5,
    xbins=dict(
        start=dur_min,
        end=dur_max,
        size=binsize
    )))
    fig11.update_layout(barmode='stack')
    fig11.update_layout(
        title_text='Modelled disturbution of run times',
        yaxis_title="Game play samples",
        xaxis_title="Time (milliseconds)")
    fig11.show()
   
    Z = machine_1.predict(x_data_1)
    new_bot = []
    new_bot_pt = []
    new_rest = []
    new_rest_pt = []
    for k,z in enumerate(Z):
        if z > 0:
            new_bot.append(duration[k])
            new_bot_pt.append(x_data_1[k])
        else:
            new_rest.append(duration[k])
            new_rest_pt.append(x_data_1[k])    
    fig12 = go.Figure()
    fig12.add_trace(go.Scatter3d(y=[x[1] for x in new_bot_pt],x=[x[0] for x in new_bot_pt],z=[x[2] for x in new_bot_pt], mode='markers',name = 'Fastest'))
    fig12.add_trace(go.Scatter3d(y=[x[1] for x in new_rest_pt],x=[x[0] for x in new_rest_pt],z=[x[2] for x in new_rest_pt], mode='markers',name = 'Rest'))
    fig12.update_layout(title_text='Modelled projected expert non-expert split', scene = dict(
                xaxis_title='Sprinting ratio',
                yaxis_title='Attacking ratio',
                zaxis_title='Median crema movment'))
    fig12.show()
            
    dur_max = max(duration)
    dur_min = min(duration)
    bins = np.linspace(dur_min, dur_max, num=64)

    binsize = bins[1] - bins[0]
    fig13 = go.Figure()
    fig13.add_trace(go.Histogram(x=new_bot, name='1',opacity=0.5,
    xbins=dict(
        start=dur_min,
        end=dur_max,
        size=binsize
    )))
    
    fig13.add_trace(go.Histogram(x=new_rest, name='0',opacity=0.5,
    xbins=dict(
        start=dur_min,
        end=dur_max,
        size=binsize
    )))
    fig13.update_layout(barmode='stack')
    fig13.update_layout(
        title_text='Modelled disturbution of run times',
        yaxis_title="Game play samples",
        xaxis_title="Time (milliseconds)")
    fig13.show()        

    
if __name__ == '__main__':
    main()
#idea -> talk over lanscapes 
#break down into indivual actions statics
# finish with abs amounts of each action and there lin kto reward
#action-time landscape
#reward-time lanscape
#absoult amount of actions taken
#courosity landscape
#indual actions vs reward
#induvual action vs time
#frame to frame ditence? 