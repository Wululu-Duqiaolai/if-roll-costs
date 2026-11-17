"""从四个账户的输出绘制论文图。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
F = R / "figures"
F.mkdir(exist_ok=True)
fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family":next((n for n in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if n in fonts),"sans-serif"),
                     "font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False})
COLORS={"month":"#29617a","quarter":"#b05331"}
NAMES={"month":"近月","quarter":"季月","daily":"每日","weekly":"每周"}

def save(fig, name):
    fig.savefig(F / (name+".png"), dpi=285, bbox_inches="tight", pad_inches=.12)
    fig.savefig(F / (name+".svg"), bbox_inches="tight", pad_inches=.12)
    plt.close(fig)

def risk_cost(s):
    fig,ax=plt.subplots(figsize=(7.3,3.8),layout="constrained")
    for cycle in COLORS:
        pair=s[(s.scenario=="base") & (s.cycle==cycle)].set_index("frequency").loc[["daily","weekly"]]
        x=pair.fee_annual_bp.to_numpy(); y=100*pair.ann_vol.to_numpy()
        ax.plot(x,y,color=COLORS[cycle],lw=1.7)
        for i,freq in enumerate(pair.index):
            ax.scatter(x[i],y[i],s=55,color=COLORS[cycle],marker=("o","D")[i],zorder=3)
            ax.annotate(f"{NAMES[cycle]}·{NAMES[freq]}  ({x[i]:.2f}, {y[i]:.2f}%)",
                        (x[i],y[i]),xytext=(8 if cycle=="quarter" else -8,6),textcoords="offset points",
                        ha="left" if cycle=="quarter" else "right",color=COLORS[cycle],fontsize=9)
    ax.axhline(15,color="#888888",ls="--",lw=1)
    ax.text(29.3,15.015,"风险预算 15%",ha="right",va="bottom",color="#696969",fontsize=9)
    ax.set(xlim=(9,30),ylim=(14.89,16.65),xlabel="年化综合交易成本（净值基点）",ylabel="扣费收益年化波动率（%）")
    ax.grid(axis="y",alpha=.18)
    save(fig,"01_risk_cost")

def thresholds(s):
    b=pd.read_csv(R/"breakeven.csv")
    stress=s[s.scenario=="far_stress"].pivot(index="frequency",columns="cycle",values="fee_annual_bp")
    fig,ax=plt.subplots(figsize=(7.3,3.65),layout="constrained")
    xs=np.linspace(1,10,120)
    ax.axhline(0,lw=1,color="#555555")
    for i,r in b.iterrows():
        c=list(COLORS.values())[i]
        ys=r.base_fee_saving_annual_bp-(xs-1)*r.incremental_far_tick_annual_bp
        ax.plot(xs,ys,label=NAMES[r.frequency]+"，固定订单",color=c,ls=("-","--")[i],lw=1.8)
        ax.scatter(r.total_far_ticks_breakeven,0,color=c,s=38,zorder=4)
        ax.annotate(f"{r.total_far_ticks_breakeven:.2f} 跳",(r.total_far_ticks_breakeven,0),
                    xytext=((-20,-19),(21,10))[i],textcoords="offset points",ha="center",color=c,fontsize=9)
        ax.scatter(9,stress.loc[r.frequency,"month"]-stress.loc[r.frequency,"quarter"],marker="x",s=52,color=c,zorder=4)
    ax.plot([],[],marker="x",ls="",color="#424242",label="9跳，重算账户")
    ax.set(xlabel="远期限腿总滑点（跳/手/单边），其他腿固定为1跳",ylabel="近月减季月的年化成本（基点）",xlim=(.8,10.2),ylim=(-9,20))
    ax.set_xticks(np.arange(1,11));ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.2)
    save(fig,"02_threshold")

def market():
    y=pd.read_csv(R/"market_year.csv").query("year!='all' and scope=='different'").copy()
    z=pd.read_csv(R/"market_liquidity.csv").query("frequency=='daily'")
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.65),layout="constrained")
    a,b=axes
    a.bar(y.year,y.spread_close_median_bp,color=[COLORS["quarter"] if val<0 else COLORS["month"] for val in y.spread_close_median_bp],width=.6)
    a.axhline(0,color="#666",lw=.8);a.set(title="不同持仓合约日的收盘价差",ylabel="季月减近月 / 现货指数（基点）")
    a.grid(axis="y",alpha=.15)
    for i,(cycle,c) in enumerate(COLORS.items()):
        rows=z[z.cycle==cycle].set_index("leg").loc[["close_old","open_new"]]
        x=np.arange(2)+(i-.5)*.22
        mid=rows.qty_pct_prior_volume_median.to_numpy(); high=rows.qty_pct_prior_volume_p95.to_numpy()
        b.vlines(x,mid,high,color=c,lw=2);b.scatter(x,mid,color=c,marker="o",s=35,label=NAMES[cycle]+" 中位数")
        b.scatter(x,high,color=c,marker="_",s=100)
    b.set(title="移仓腿对前一交易日成交量的占比",ylabel="母订单手数 / 昨日同合约成交量（%）",xticks=[0,1],xticklabels=["平旧仓","开新仓"],xlim=(-.45,1.45),ylim=(0,1.8))
    b.legend(frameon=False,fontsize=8,loc="upper left");b.grid(axis="y",alpha=.15)
    b.text(.98,.97,"短横线为第95百分位",transform=b.transAxes,ha="right",va="top",fontsize=8,color="#555555")
    save(fig,"03_basis_liquidity")

if __name__=="__main__":
    summary=pd.read_csv(R/"accounts_summary.csv")
    risk_cost(summary)
    thresholds(summary)
    market()
