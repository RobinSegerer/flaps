"""Interactive EFA/CFA teaching modules for FLAPs; continuous Gaussian example."""
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2

ITEMS = [f'I{i}' for i in range(1, 9)]
TEXTS = ['Feste Lernzeiten planen', 'Aufgaben in Schritte teilen', 'Lernziele setzen',
         'Aufgaben priorisieren', 'Ablenkungen widerstehen', 'Bei Schwierigkeiten weiterarbeiten',
         'Konzentriert bleiben', 'Ohne Aufschieben beginnen']


def validate_matrix(r):
    r = np.asarray(r, float)
    if r.shape != (8, 8) or not np.isfinite(r).all():
        raise ValueError('Bitte acht vollständige Zeilen und Spalten mit Zahlen eingeben.')
    if not np.allclose(r, r.T) or not np.allclose(np.diag(r), 1):
        raise ValueError('Die Korrelationsmatrix muss symmetrisch sein und Einsen auf der Diagonale haben.')
    if np.max(np.abs(r)) > 1 or np.linalg.eigvalsh(r).min() <= 1e-6:
        raise ValueError('Diese Matrix ist nicht positiv definit. Bitte die Zusammenhänge anpassen; sie wird nicht automatisch verändert.')
    return r


def diagnostics(r, n):
    precision = np.linalg.inv(r)
    partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(partial, 0)
    off = r.copy(); np.fill_diagonal(off, 0)
    a, b = (off**2).sum(axis=0), (partial**2).sum(axis=0)
    msa = np.divide(a, a+b, out=np.full(len(a), np.nan), where=a+b>1e-14)
    kmo = a.sum()/(a+b).sum() if (a+b).sum()>1e-14 else np.nan
    p = len(r); df = p*(p-1)//2
    stat = max(0., -(n-1-(2*p+5)/6)*np.linalg.slogdet(r)[1])
    return kmo, msa, stat, df, chi2.sf(stat, df)


def paf(r, factors, tolerance=1e-6, max_iter=300):
    h = 1-1/np.diag(np.linalg.inv(r))
    history = [h.copy()]
    for iteration in range(max_iter):
        reduced = r.copy(); np.fill_diagonal(reduced, h)
        values, vectors = np.linalg.eigh(reduced)
        order = np.argsort(values)[::-1][:factors]
        if np.any(values[order] <= 0):
            raise ValueError('Für diese Faktorenanzahl fehlen positive Eigenwerte der reduzierten Matrix.')
        loadings = vectors[:, order]*np.sqrt(values[order])
        new = (loadings**2).sum(axis=1)
        history.append(new.copy())
        if np.any(new>1+1e-6):
            return loadings, history, 'Heywood-Fall: Eine Kommunalität übersteigt 1. Keine reguläre Lösung.'
        if np.max(np.abs(new-h)) < tolerance:
            return loadings, history, None
        h = new
    return loadings, history, 'Die PAF hat innerhalb von 300 Iterationen nicht konvergiert.'


def matrix_view(matrix, title, zlim=1):
    fig = go.Figure(go.Heatmap(z=matrix, x=ITEMS, y=ITEMS, zmin=-zlim, zmax=zlim,
        colorscale='RdBu', reversescale=True, texttemplate='%{z:.2f}'))
    fig.update_layout(title=title, height=390, margin=dict(t=45,b=10), yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)


def generated_data(within=.6, rho=.3, first=.75, n=500, seed=17):
    groups = np.array([0]*4+[1]*4)
    lam = np.zeros((8,2)); lam[np.arange(8),groups]=np.sqrt(within)
    lam[0,0] *= first
    phi = np.array([[1,rho],[rho,1]])
    common = lam@phi@lam.T
    population = common+np.diag(1-np.diag(common))
    data = np.random.default_rng(seed).normal(size=(n,8))@np.linalg.cholesky(population).T
    return np.corrcoef(data,rowvar=False)


def efa_data():
    with st.expander('Die acht Lernkompetenzitems'):
        st.dataframe(pd.DataFrame({'Item':ITEMS,'Inhalt':TEXTS,'Facette':['Lernplanung']*4+['Selbstkontrolle']*4}),hide_index=True)
    st.caption('Fiktive kontinuierliche Itemwerte. I1–I4 betreffen Lernplanung, I5–I8 Selbstkontrolle. Keine validierte Skala; ordinale Antworten erfordern gegebenenfalls andere Methoden.')
    n = st.select_slider('Stichprobengröße N', [100,200,500,1000,2000], value=500, key='efa_n')
    within = st.slider('Zusammenhang innerhalb der Facetten', .2,.8,.6,.05,key='efa_within')
    rho = st.slider('Korrelation der Facetten', 0.,1.,.3,.05,key='efa_rho')
    first = st.slider('Relative Ladungsstärke von Item 1', .3,1.,.75,.05,key='efa_first')
    if 'efa_seed' not in st.session_state: st.session_state.efa_seed=17
    if st.button('Neue Stichprobe ziehen',key='efa_new'): st.session_state.efa_seed+=1
    r = generated_data(within,rho,first,n,st.session_state.efa_seed)
    manual = st.checkbox('Korrelationsmatrix selbst bearbeiten',key='efa_manual')
    if manual:
        st.caption('Nur Werte oberhalb der Diagonale werden übernommen und gespiegelt. N wird für Tests und Parallelanalyse als angenommene Stichprobengröße verwendet; zu einer frei bearbeiteten Matrix liegen keine Rohdaten vor.')
        edited = st.data_editor(pd.DataFrame(r,index=ITEMS,columns=ITEMS),key='efa_editor',use_container_width=True)
        upper = np.triu(edited.to_numpy(dtype=float),1)
        r = upper+upper.T+np.eye(8)
    try: r=validate_matrix(r)
    except ValueError as error:
        st.error(str(error)); return None
    with st.expander('Itemstandardabweichungen für Kovarianzen, CFA und Reliabilität'):
        st.caption('Die EFA arbeitet mit Korrelationen. Aus diesen Standardabweichungen entsteht die Kovarianzmatrix für CFA und Alpha/Omega. Verschiedene Skalierungen beeinflussen den ungewichteten Summenscore.')
        scales = st.data_editor(pd.DataFrame({'Item':ITEMS,'SD':[1.]*8}),disabled=['Item'],hide_index=True,key='efa_sds')
        sd = scales['SD'].to_numpy(dtype=float)
    if not np.isfinite(sd).all() or np.any(sd<=0):
        st.error('Alle Standardabweichungen müssen positiv und endlich sein.'); return None
    payload = dict(r=r, covariance=r*np.outer(sd,sd), n=n, manual=manual)
    st.session_state.fa_data=payload
    return payload


@st.cache_data(show_spinner=False)
def parallel_eigenvalues(n, seed=152, repetitions=100):
    rng=np.random.default_rng(seed)
    return np.quantile([np.linalg.eigvalsh(np.corrcoef(rng.normal(size=(n,8)),rowvar=False))[::-1]
                       for _ in range(repetitions)],.95,axis=0)


def module_efa():
    st.title('8 · Exploratorische Faktorenanalyse')
    st.write('Welche gemeinsamen Dimensionen erklären die Zusammenhänge zwischen den Lernkompetenzitems? Erkunden Sie zuerst die Matrix, dann die Hauptachsenanalyse (PAF). Dieselbe Datengrundlage steht anschließend in Modul 9 bereit.')
    payload=efa_data()
    if payload is None: return
    r,n=payload['r'],payload['n']
    matrix_view(r,'Empirische Korrelationsmatrix')
    kmo,msa,stat,df,p=diagnostics(r,n)
    c1,c2,c3=st.columns(3)
    c1.metric('KMO gesamt',f'{kmo:.3f}' if np.isfinite(kmo) else 'nicht bestimmbar')
    c2.metric(f'Bartlett χ² (df = {df})',f'{stat:.2f}')
    c3.metric('Bartlett p',f'{p:.3g}')
    st.dataframe(pd.DataFrame({'Item':ITEMS,'MSA':np.round(msa,3)}),hide_index=True)
    st.caption('KMO/MSA vergleichen einfache mit partiellen Korrelationen: größere Werte sprechen für ein günstigeres gemeinsames Zusammenhangsmuster. Bartlett prüft die Einheitsmatrix und hängt von N ab. Signifikanz bestimmt weder die Faktorenanzahl noch garantiert sie eine brauchbare Lösung. Näherung für unabhängige, multivariat normalverteilte Beobachtungen.')
    with st.expander('Screeplot und Parallelanalyse'):
        eigen=np.linalg.eigvalsh(r)[::-1]; ref=parallel_eigenvalues(n)
        fig=go.Figure()
        fig.add_scatter(x=list(range(1,9)),y=eigen,name='Eigenwerte der vollständigen R',mode='lines+markers')
        fig.add_scatter(x=list(range(1,9)),y=ref,name='95. Perzentil · 100 Zufallsmatrizen',mode='lines+markers')
        fig.update_layout(xaxis_title='Rang des Eigenwerts',yaxis_title='Eigenwert',height=330)
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Komponentenbasierte Parallelanalyse der vollständigen Korrelationsmatrix als Orientierung. Dies sind keine Eigenwerte der reduzierten PAF-Matrix. Die Faktorenanzahl zusätzlich anhand Theorie, Ladungen und Residuen beurteilen; die Linie ist keine automatische Entscheidung.')
    factors=st.radio('Anzahl zu extrahierender Faktoren',[1,2,3],horizontal=True,key='efa_factors')
    rotation=st.selectbox('Rotation',['Keine','Varimax · unkorrelierte Faktoren','Promax · korrelierte Faktoren'],key='efa_rotation',disabled=factors==1)
    try: loadings,history,problem=paf(r,factors)
    except ValueError as error:
        st.error(str(error)); return
    if problem: st.warning(problem)
    with st.expander('PAF Schritt für Schritt'):
        step=st.select_slider('Iterationsschritt',options=list(range(len(history))),value=len(history)-1,key='efa_step')
        st.write('Schritt 0: Startkommunalitäten aus quadrierten multiplen Korrelationen (SMC). Danach ersetzen wir die Diagonale durch die aktuellen Kommunalitäten, extrahieren Faktoren und berechnen neue Kommunalitäten. Abbruch bei maximaler Änderung < 0,000001.')
        st.dataframe(pd.DataFrame({'Item':ITEMS,'Kommunalität':history[step]}).round(4),hide_index=True)
        reduced=r.copy(); np.fill_diagonal(reduced,history[step])
        matrix_view(reduced,f'Reduzierte Matrix nach Schritt {step}')
        st.caption('Die Ergebnistabellen unten zeigen die letzte berechnete Lösung. Die Diagonale der reduzierten Matrix beschreibt geschätzte gemeinsame Varianz, keine vollständige Itemvarianz.')
    if problem: return
    phi=np.eye(factors)
    if factors>1 and rotation!='Keine':
        from factor_analyzer.rotator import Rotator
        rot=Rotator(method='varimax' if rotation.startswith('Varimax') else 'promax')
        loadings=rot.fit_transform(loadings)
        if rot.phi_ is not None: phi=rot.phi_
    common=loadings@phi@loadings.T
    h=np.diag(common)
    table=pd.DataFrame(loadings,index=ITEMS,columns=[f'F{i+1}' for i in range(factors)])
    table['h²']=h; table['1 − h²']=1-h
    st.dataframe(table.round(3),use_container_width=True)
    st.caption('Ladungen sind Musterkoeffizienten. Bei korrelierten Faktoren gilt h² = Diagonale von ΛΦΛᵀ; die bloße Summe quadrierter Ladungen reicht dann nicht. Rotation verändert die reproduzierten Zusammenhänge nicht. Faktorvorzeichen und Reihenfolge sind nicht inhaltlich festgelegt.')
    if factors>1:
        st.write('Faktorenkorrelationen'); st.dataframe(pd.DataFrame(phi,index=table.columns[:factors],columns=table.columns[:factors]).round(3))
        if rotation.startswith('Promax'):
            with st.expander('Strukturkoeffizienten: Korrelationen der Items mit den Faktoren'):
                st.dataframe(pd.DataFrame(loadings@phi,index=ITEMS,columns=table.columns[:factors]).round(3))
    reproduced=common+np.diag(1-h)
    st.metric('RMSR · Abweichungen außerhalb der Diagonale',f'{np.sqrt(np.mean((r-reproduced)[np.triu_indices(8,1)]**2)):.3f}')
    c1,c2=st.columns(2)
    with c1: matrix_view(reproduced,'Reproduzierte Korrelationen')
    with c2: matrix_view(r-reproduced,'Empirisch minus reproduziert')
    st.info('Ausprobieren: einen statt zwei Faktoren extrahieren; danach Varimax und Promax vergleichen. Welche Zuordnung passt zum Inhalt? Die Simulationseinstellungen bestimmen die Daten, die gewählte Faktorenanzahl bestimmt Ihre Analyse.')
    st.markdown('Methodik: [PAF im psych-Paket](https://personality-project.org/r/html/fa.html) · [Rotationsverfahren](https://factor-analyzer.readthedocs.io/en/latest/introduction.html).')


def cfa_description(groups, scaling='unit', equal=False, correlated=True, cross=(), residual=()):
    dimensions=max(groups)+1; factors=[f'F{i+1}' for i in range(dimensions)]
    lines=['DEFINE(latent) '+' '.join(factors)]
    starts=[]
    for factor in range(dimensions):
        members=[i for i,g in enumerate(groups) if g==factor]
        anchor=members[0]
        for i in members:
            parameter = ('1' if scaling=='marker' and (equal or i==anchor) else
                         f'equal{factor}' if equal else f'l{i}')
            lines.append(f'{ITEMS[i]} ~ {parameter}*{factors[factor]}')
            if parameter!='1': starts.append(parameter)
        lines.append(f'{factors[factor]} ~~ '+('1*' if scaling=='unit' else '')+factors[factor])
    for i in cross:
        other=1-groups[i]; parameter=f'cross{i}'
        lines.append(f'{ITEMS[i]} ~ {parameter}*{factors[other]}'); starts.append(parameter)
    for i,j in residual: lines.append(f'{ITEMS[i]} ~~ {ITEMS[j]}')
    if dimensions==2: lines.append('F1 ~~ '+('' if correlated else '0*')+'F2')
    if starts: lines.append('START(0.7) '+' '.join(sorted(set(starts))))
    return '\n'.join(lines)


def fit_cfa(covariance, n, description):
    from semopy import Model
    model=Model(description)
    opt=model.fit(cov=pd.DataFrame(covariance,index=ITEMS,columns=ITEMS),n_samples=n,obj='MLW',options={'maxiter':1500,'ftol':1e-10})
    if not opt.success: raise ValueError('Die Parameterschätzung hat nicht konvergiert.')
    sigma,(m,c)=model.calc_sigma()
    observed=list(model.vars['observed'])
    order=[observed.index(item) for item in ITEMS]
    sigma=sigma[np.ix_(order,order)]
    jac=np.column_stack([g[np.triu_indices(8)] for g in model.calc_sigma_grad(m,c)])
    norms=np.linalg.norm(jac,axis=0)
    rank=np.linalg.matrix_rank(jac/np.maximum(norms,1e-15),tol=1e-7)
    free=len(model.param_vals); df=36-free
    if rank<free or df<0: raise ValueError(f'Modell lokal nicht identifiziert: {free} freie Parameter, aber nur {rank} unabhängige Ableitungsrichtungen. Einschränkungen ergänzen.')
    estimates=model.inspect(std_est=True)
    for column in estimates.columns:
        if column not in ['lval','op','rval']:
            estimates[column]=pd.to_numeric(estimates[column],errors='coerce')
    matrices=model.inspect(mode='mx')
    lam=matrices['Lambda'].loc[ITEMS].to_numpy()
    phi=matrices['Psi'].to_numpy(); theta=matrices['Theta'].loc[ITEMS,ITEMS].to_numpy()
    if np.linalg.eigvalsh(phi).min()<=0 or np.linalg.eigvalsh(theta).min()<=1e-8:
        raise ValueError('Unzulässige oder grenzwertige Lösung: Faktor- oder Residualkovarianzmatrix ist nicht positiv definit.')
    discrepancy=np.linalg.slogdet(sigma)[1]+np.trace(np.linalg.solve(sigma,covariance))-np.linalg.slogdet(covariance)[1]-8
    stat=max(0.,n*discrepancy)
    base=max(0.,n*(np.log(np.diag(covariance)).sum()-np.linalg.slogdet(covariance)[1]))
    excess=max(stat-df,0.); base_excess=base-28
    scaling=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
    srmr=np.sqrt(np.mean(((covariance-sigma)/scaling)[np.triu_indices(8)]**2))
    alpha=8/7*(1-np.trace(covariance)/covariance.sum())
    omega=(lam@phi@lam.T).sum()/sigma.sum()
    factor_corr=phi/np.sqrt(np.outer(np.diag(phi),np.diag(phi)))
    return dict(sigma=sigma, estimates=estimates, lam=lam, phi=phi, theta=theta,
                factor_corr=factor_corr, free=free, df=df, chi2=stat, p=chi2.sf(stat,df) if df>0 else np.nan,
                RMSEA=np.sqrt(excess/(df*(n-1))) if df>0 else np.nan,
                CFI=1-excess/max(excess,base_excess) if df>0 and base_excess>1e-10 else np.nan,
                TLI=(base/28-stat/df)/(base/28-1) if df>0 and base_excess>1e-10 else np.nan,
                SRMR=srmr, alpha=alpha, omega=omega)


def cfa_diagram(result, groups):
    fig=go.Figure(); dimensions=max(groups)+1
    fx=[3.5] if dimensions==1 else [1.5,5.5]
    for f,x in enumerate(fx):
        fig.add_shape(type='circle',x0=x-.5,x1=x+.5,y0=1.8,y1=2.2,line_color='#357ca5')
        fig.add_annotation(x=x,y=2,text=f'F{f+1}',showarrow=False)
    sd=np.sqrt(np.diag(result['sigma']))
    for i in range(8):
        fig.add_annotation(x=i,y=0,text=ITEMS[i],showarrow=False,bgcolor='#eee',borderpad=10)
        for f,x in enumerate(fx):
            loading=result['lam'][i,f]*np.sqrt(result['phi'][f,f])/sd[i]
            if abs(loading)>1e-10:
                fig.add_annotation(x=i,y=.15,ax=x,ay=1.8,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=2,arrowcolor='#357ca5')
                fig.add_annotation(x=.65*i+.35*x,y=.7,text=f'{loading:.2f}',showarrow=False,bgcolor='white')
    if dimensions==2:
        fig.add_annotation(x=5.,y=2,ax=2.,ay=2,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowside='end+start',arrowhead=2,startarrowhead=2)
        fig.add_annotation(x=3.5,y=2.25,text=f"r = {result['factor_corr'][0,1]:.2f}",showarrow=False)
    fig.update_layout(height=330,margin=dict(t=15,b=10),xaxis=dict(visible=False,range=[-.5,7.5]),yaxis=dict(visible=False,range=[-.3,2.5]))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Standardisierte Ladungen und Faktorenkorrelation. Residualvarianzen und zugelassene Residualkovarianzen stehen in der Parametertabelle.')


def module_cfa():
    st.title('9 · Konfirmatorische Faktorenanalyse')
    st.write('Formulieren Sie eine Annahme über dieselben Lernkompetenzitems: Welche Items messen welchen Faktor? Vergleichen Sie alternative Modelle an derselben Kovarianzmatrix.')
    if 'fa_data' not in st.session_state:
        r=generated_data(); st.session_state.fa_data=dict(r=r,covariance=r,n=500,manual=False)
    payload=st.session_state.fa_data; cov=payload['covariance']; n=payload['n']
    st.caption(f'Datengrundlage aus Modul 8 · N = {n}. Ohne vorherigen Besuch der EFA wird das Standardbeispiel verwendet. Itemstandardabweichungen und Matrix können in Modul 8 verändert werden.')
    dimensions=st.radio('Anzahl Faktoren',[1,2],index=1,horizontal=True,key='cfa_dimensions')
    groups=[0]*8
    if dimensions==2:
        first=st.multiselect('Items für Faktor 1; übrige Items gehören zu Faktor 2',ITEMS,default=ITEMS[:4],key='cfa_groups')
        if not 3<=len(first)<=5:
            st.info('Bitte mindestens drei primäre Items je Faktor wählen.'); return
        groups=[0 if item in first else 1 for item in ITEMS]
    scaling=st.radio('Skalierung',['Faktorvarianz = 1','Referenzladung = 1'],horizontal=True,key='cfa_scaling')
    equal=st.checkbox('Primäre Ladungen innerhalb jedes Faktors gleichsetzen',key='cfa_equal')
    correlated=st.checkbox('Faktoren dürfen korrelieren',value=True,key='cfa_correlated',disabled=dimensions==1)
    with st.expander('Zusätzliche Parameter freigeben'):
        cross_items=st.multiselect('Kreuzladungen: Diese Items laden zusätzlich auf dem anderen Faktor',ITEMS,key='cfa_cross',disabled=dimensions==1)
        pairs=[f'{ITEMS[i]} ↔ {ITEMS[j]}' for i in range(8) for j in range(i+1,8)]
        selected=st.multiselect('Residualkovarianzen',pairs,key='cfa_residual')
        st.caption('Parameter nur mit inhaltlicher Begründung freigeben. Die Zahl der Freiheitsgrade allein garantiert keine Identifikation; zusätzlich wird der lokale Rang der Modellableitungen geprüft.')
    cross=[ITEMS.index(item) for item in cross_items] if dimensions==2 else []
    residual=[tuple(ITEMS.index(item) for item in pair.split(' ↔ ')) for pair in selected]
    desc=cfa_description(groups,'unit' if scaling.startswith('Faktor') else 'marker',equal,correlated,cross,residual)
    fingerprint=hashlib.sha256(cov.tobytes()+str(n).encode()).hexdigest()
    if st.session_state.get('cfa_fingerprint')!=fingerprint:
        st.session_state.cfa_fingerprint=fingerprint; st.session_state.cfa_history=[]
    signature=(fingerprint,desc)
    if st.button('CFA schätzen',type='primary',key='cfa_fit'):
        try:
            result=fit_cfa(cov,n,desc)
            st.session_state.cfa_result=(signature,result)
            row={'Modell':f'{dimensions} F · '+('gleich' if equal else 'freie Ladungen')+f' · {len(cross)} Kreuzl. · {len(residual)} Restkov.',
                 'Skalierung':scaling,'Faktoren korreliert':bool(dimensions==2 and correlated),
                 **{key:result[key] for key in ['chi2','df','RMSEA','CFI','TLI','SRMR','alpha','omega']}}
            st.session_state.cfa_history.append(row)
        except (ValueError,np.linalg.LinAlgError) as error:
            st.session_state.pop('cfa_result',None); st.error(str(error))
    saved=st.session_state.get('cfa_result')
    if saved and saved[0]==signature:
        result=saved[1]
        cfa_diagram(result,groups)
        cols=st.columns(5)
        cols[0].metric(f"χ² (df = {result['df']})",f"{result['chi2']:.2f}")
        for col,key in zip(cols[1:],['RMSEA','CFI','TLI','SRMR']):
            col.metric(key,f'{result[key]:.3f}' if np.isfinite(result[key]) else '—')
        st.caption(f"36 empirische Varianzen/Kovarianzen − {result['free']} freie Parameter = {result['df']} Freiheitsgrade. p = {result['p']:.3g}. Normaltheoretische ML-Schätzung ohne Mittelwertstruktur; keine robusten Fitindizes.")
        if result['df']==0: st.warning('Gesättigtes Modell: Der perfekte Fit ist kein empirischer Modelltest.')
        if dimensions==2 and abs(result['factor_corr'][0,1])>.95: st.warning('Die Faktoren sind sehr hoch korreliert. Ihre empirische Unterscheidbarkeit ist eingeschränkt.')
        st.dataframe(result['estimates'].rename(columns={'Est. Std':'Standardisiert'}),hide_index=True,use_container_width=True)
        with st.expander('Empirische und modellierte Kovarianzen vergleichen'):
            for title,matrix in [('Empirisch',cov),('Modellimpliziert',result['sigma']),('Abweichungen',cov-result['sigma'])]:
                st.write(title);st.dataframe(pd.DataFrame(matrix,index=ITEMS,columns=ITEMS).round(3))
        st.subheader('Cronbachs Alpha und McDonalds Omega')
        a,b=st.columns(2); a.metric('α · empirische Kovarianzen',f"{result['alpha']:.3f}")
        b.metric('ω · alle gemeinsamen Faktoren',f"{result['omega']:.3f}")
        st.latex(r'\alpha=\frac{k}{k-1}\left(1-\frac{\sum_i Var(X_i)}{Var(\sum_iX_i)}\right)')
        st.latex(r'\omega=\frac{\mathbf{1}^{\mathsf T}\Lambda\Phi\Lambda^{\mathsf T}\mathbf{1}}{\mathbf{1}^{\mathsf T}(\Lambda\Phi\Lambda^{\mathsf T}+\Theta)\mathbf{1}}')
        st.write('Beide Werte beziehen sich auf den ungewichteten Summenscore in der eingegebenen Skalierung. Alpha wird aus der empirischen Kovarianzmatrix berechnet. Omega verwendet die modellierte gemeinsame Varianz und Gesamtvarianz. Residualkovarianzen werden hier als nicht zum Zielmerkmal gehörende gemeinsame Restanteile behandelt und im Nenner berücksichtigt.')
        if dimensions==1:
            st.latex(r'\omega=\frac{(\sum_i\lambda_i)^2 Var(\eta)}{(\sum_i\lambda_i)^2 Var(\eta)+\sum_i Var(\varepsilon_i)+2\sum_{i<j}Cov(\varepsilon_i,\varepsilon_j)}')
        else:
            st.info('Dieses Omega zählt beide Faktoren als wahre gemeinsame Varianz. Es ist kein Omega hierarchisch und belegt keinen eindimensionalen Gesamtscore. Ob ein Gesamtwert sinnvoll ist, bleibt eine inhaltliche Frage.')
        st.caption('Für den Alpha–Omega-Vergleich zuerst ein eindimensionales Beispiel wählen (in Modul 8 Facettenkorrelation = 1). Dann Item 1 abschwächen und gleiche mit freien Ladungen vergleichen. Bei passendem tau-äquivalentem Modell mit unkorrelierten Fehlern stimmen beide in der Population überein. Eine schlecht passende CFA liefert keine verlässliche Grundlage für Omega.')
    elif saved:
        st.info('Die Modellspezifikation wurde verändert. Bitte neu schätzen; alte Ergebnisse werden nicht als aktuelle Lösung angezeigt.')
    if st.session_state.get('cfa_history'):
        st.subheader('Modelle an derselben Datengrundlage vergleichen')
        st.dataframe(pd.DataFrame(st.session_state.cfa_history).round(3),hide_index=True,use_container_width=True)
        st.caption('Kleinere Residuen und günstigere Fitwerte gemeinsam mit Theorie und Sparsamkeit beurteilen. Mehr freie Parameter verbessern den Fit häufig. Die Tabelle ist kein automatischer χ²-Differenztest. Beim Ändern der Datengrundlage wird sie geleert.')
    with st.expander('Modellspezifikation und methodischer Hintergrund'):
        st.code(desc,language='text')
        st.markdown('[semopy: Syntax](https://semopy.com/syntax.html) · [Schätzung und Fit](https://semopy.com/tutorial.html). Die Eingabematrix wird als Kovarianzmatrix mit ML-Konvention behandelt; χ² = N · FML. SRMR mittelt quadrierte standardisierte Kovarianzabweichungen einschließlich Diagonale. KMO und Bartlett gehören zur vorbereitenden Matrixprüfung in Modul 8.')
