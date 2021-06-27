### Linear SVM ###

suppose there are n data, m dimensions

m=2 dimensions
$$
\begin{align}
p&=(y截距之差*cos\theta)=(\frac{b_u}{w_2}-\frac{b_d}{w_2})cos\theta 
\Rightarrow cos\theta=\frac{pw_2}{b_u-b_d}\\
&=(x截距之差*sin\theta)=(\frac{b_u}{w_1}-\frac{b_d}{w_1})sin\theta 
\Rightarrow sin\theta=\frac{pw_1}{b_u-b_d}\\
\because &sin^2\theta+cos^2\theta=1 \\
&\Rightarrow  (\frac{pw_2}{b_u-b_d})^2+(\frac{pw_1}{b_u-b_d})^2=1 \\
\therefore &p=\frac{b_u-b_d}{\sqrt{w_1^2+w_2^2}} \\
\because &\text{we hope that the path p can be the widest} \\
&\Rightarrow \text{maximize p}\\
&\Rightarrow \text{minimize }w_1^2+w_2^2 \\
s.t. &\sum_{j=1}^2 w_jx_j-b_u>0 \text{ if loan good}\\
&\sum_{j=1}^2 w_jx_j-b_d<0 \text{ if loan bad}\\
\text{suppose }&b_u=b+1,b_d=b-1\\
&\Rightarrow \text{maximize }p=\frac{2}{\sqrt{w_1^2+w_2^2}} \\
&\Rightarrow \text{minimize }w_1^2+w_2^2 \\
s.t. &\sum_{j=1}^2 w_jx_j>b+1 \text{ if loan good}\\
&\sum_{j=1}^2 w_jx_j<b-1 \text{ if loan bad}\\
\end{align}
$$

![simpleSVM](./simpleSVM.jpeg)

m dimensions
$$
min_{\{\vec w,b\}}\sum_{j=1}^m w_j^2\\
\begin{align}
s.t. 
&\sum_{j=1}^mw_jx_j-b > 1\text{ if loan good}\\
&\sum_{j=1}^mw_jx_j-b < -1\text{ if loan bad}
\end{align}
$$


hint : 可以使用二次規劃(quadratic programming) 來解出$\vec w$ 與$b$，gurobi solver有提供套件可使用



### Non-linear SVM ###

如果data $x_i$在原本的feature space裡很難被線性劃分的話，那麼我們可以把這些data $x_i$非線性轉換$\phi(x_i)$至可以被線性劃分的data $z_i$ (primal problem)
$$
z_i=\phi(x_i) , i=1,...,n\\
min _{\{\vec w,b\}}\sum_{j=1}^m w_j^2\\
\begin{align}
s.t. 
&\sum_{j=1}^mw_jz_j-b > 1\text{ if loan good}\\
&\sum_{j=1}^mw_jz_j-b < -1\text{ if loan bad}
\end{align}
$$
但若是特徵轉換函式$\phi(x_i)$非常複雜，計算複雜度就會被$\phi(x_i)$給拉大，因此有人想出了利用primal-dual problem的技巧來使用kernel function簡化透過$\phi(x_i)$將data $x_i$特徵轉換至data $z_i$的過程，使得計算複雜度只跟data數量有關

Largrange multiplier‘s original version

> $$
> \begin{align}
> &\text{min/max}_{w} f(w)\\
> &\text{s.t. } g(w)=c\\
> &\text{當極值發生時, }f(w)=d_n\text{與}g(w)=c\text{相切, 即兩式的切線在此點平行}\\
> \Rightarrow& \text{兩式的法向量在此點平行} \\
> &\nabla_w f(w)=-\lambda \nabla_w(g(w)-c)\\
> & \nabla_w [f(w)+\lambda (g(w)-c)]=0\\
> &\text{令}L(w,\lambda):=f(w)+\lambda (g(w)-c)\\
> 
> \Rightarrow& 則可將原本的梯度函式改寫成 \\
> &\nabla_w L(w,\lambda)=\frac{\partial L(w,\lambda)}{\partial w}=\nabla_w f(w)+\lambda \nabla_w g(w)=0\\
> 
> \Rightarrow&\text{又為了要滿足限制式 }g(w)=c,需令\\
> 
> & \nabla_\lambda L(w,\lambda)=\frac{\partial L(w,\lambda)}{\partial \lambda}=g(w)-c=0\\
> &\text{hence, } \text{min/max}_{w} L(w,\lambda)=\text{min/max}_{w}f(w)\\
> &\because L(w,\lambda)\text{ reach its extreme value when }g(w)-c=0
> \end{align}
> $$
> 
> ※ 其實我們可以把限制式的常數c移項至限制式的左式裡，這樣限制式的右式就會是0

recall economic problem using Lagrangian multiplier

> 效用函數 : f(x,y)=xy
>
> 預算限制 : g(x,y)=20x+10y=600
> $$
> \begin{align}
> &L(x,y,\lambda)=xy+\lambda (20x+10y-600)\\
> \Rightarrow&\nabla L(x,y,\lambda)=0
> \begin{cases}
> \frac{\partial L}{\partial x}=0 \Rightarrow y-20\lambda=0\Rightarrow y=20\lambda\\
> \frac{\partial L}{\partial y}=0 \Rightarrow x-10\lambda=0\Rightarrow x=10\lambda\\
> \frac{\partial L}{\partial \lambda}=0 \Rightarrow 20x+10y-600=0
> \end{cases}\\
> \Rightarrow& 20(10\lambda)+10(20\lambda)-600=0, \lambda=1.5\\
> \Rightarrow& x=15,y=30
> \end{align}
> $$
> <!-- <img src="./EconomicExample.mov" alt="EconomicExample" style="zoom:20%;" /> -->

Largrange multiplier‘s general version

> $$
> \text{primal optimization problem}:\\
> min_w f(w)\\
> s.t. g_i(w) \leq0, i=1,...,k\\
> h_i(w)=0, i=1,...,l\\\\
> \text{dual generalized Lagrangian}:\\
> L(w,\alpha,\beta)=f(w)+\sum_{i=1}^k\alpha_ig_i(w)+\sum_{i=1}^l\beta_ih_i(w)\\
> $$
> ~~\Rightarrow min_w(max_{\alpha,\beta}L(w,\alpha,\beta))​~~
>
> 若希望primal problem與dual problem有相同的最佳解，需滿足KKT條件，以下會解釋



首先透過Lagrangian function來得到無限制式的版本(dual problem)，並透過$y_i$將限制式簡化成一條

$$
\Rightarrow min_{\vec w,b}(max_{\alpha}[L(\vec w,b,\alpha)=\sum_{j=1}^m\frac{1}{2}w_j^2+\sum_{i=1}^n\alpha_i*[1-y_i*(\vec w^Tz_i+b)]])\\
\left\{
\begin{align}
y_i=1 \text{ if loan good}
\\
y_i=-1 \text{ if loan bad}
\end{align}
\right.
$$

並透過以下限制式(通稱KKT (Karush-Kuhn-Tucker) Conditions)來使得primal problem的最佳解與dual problem的最佳解相同

1. Primal Feasibility Condition $g_i(w)\leq0$ :  

   > 希望最終出來的解滿足不等式限制式 $g_i(w) \leq 0$

$$
1-y_i*(\vec w^Tz_i+b) \leq0
$$


2. Dual Feasibility Condition $\alpha_i\geq0$ : 

   > 根據first order Taylor expansion
   >
   > $func(\theta)\sim func(\theta_0)+(\theta-\theta_0)\nabla func(\theta_0)$
   >
   > 當希望結果是局部"下降"時
   >
   > $func(\theta)-func(\theta_0) \sim (\theta-\theta_0)\nabla func(\theta_0)<0$
   >
   > 當希望下一步的方向可以讓下降幅度最大時
   >
   > $(\theta-\theta_0)\nabla func(\theta_0)=|(\theta-\theta_0)||\nabla func(\theta_0)|cos(兩向量的夾角)$
   >
   > $cos(兩向量的夾角)=-1$，也就是$(\theta-\theta_0)$與$\nabla func(\theta_0)$互為反方向時可以得到$(\theta-\theta_0)\nabla func(\theta_0)$的極小值
   >
   > 則可以得出$(\theta-\theta_0)=-\nabla func(\theta_0)*\eta, \eta>0$，也就是我們常看到的 $\theta=\theta_0-\eta \nabla f(\theta_0)$，總結出局部下降最快的方向為梯度的反方向（反過來可說局部上升最快的方向為梯度方向）
   >
   > 
   >
   > 對於滿足限制式條件的最佳解$w$上，$f(w)$與$g_i(w)$相切，因此存在$\alpha$使得$\nabla f(w)=-\alpha_i\nabla g_i(w)$
   >
   > 因為最終要求的是極"小"值$\text{min }_wf(w)$，表示可行解區域$g_i(w)<0$的任何一個解都比最佳解$\text{min }_wf(w)$大，又梯度方向指的是局部"上升"最快的方向，所以在最佳解上的$\nabla f(w)$的方向會指向可行解區域$g_i(w)<0$內（內部解比最佳解大）
   >
   > 但由於$\nabla g_i(w)$恆指向非可行解區域$g_i(w)>0$的地方（同樣用到梯度方向指向"上升"最快的方向的概念），
   >
   > <!--<img src="./法向量與大於零的區域.mov" alt="法向量與大於零的區域" style="zoom:80%;" />-->
   >
   > 因此必須使$\alpha_i\geq0$才能使$\nabla f(w)$指向$\nabla g_i(w)$所指的反方向，並讓最佳解上的$\nabla f(w)$的方向可以指向可行解區域$g_i(w)<0$內
   >
   > 相反地，若最終要求的是極"大"值$\text{max }_wf(w)$，則$\nabla f(w)$的方向需往非可行解區域$g_i(w)>0$走，因此必須使$\alpha_i \leq 0$

$$
\alpha_i\geq0
$$

3. Complementary Slackness Condition $\alpha_ig_i(w)=0$ : 

   > 滿足限制式條件的最佳解$w$可能會位於$g_i(w)<0$或$g_i(w)=0$
   >
   > 1. 如果最佳解$w$壓到了邊界,  $g_i(w)=0$
   >
   > 2. 如果最佳解$w$沒壓到邊界, $g_i(w)<0$，則原問題可以直接看成沒有限制式的版本$min_wf(w)$，直接令$\nabla f(w)=0$即可，同時$\nabla f(w)=-\alpha_i \nabla g_i(w)=0$得出此時$\alpha_i=0$
   >
   > $\Rightarrow$無論有沒有壓到邊界，$\alpha_ig_i(w)=0$

$$
\begin{align}
&\alpha_i*[1-y_i*(\vec w^Tz_i+b)]=0\\
&\text{which means that }\\
&\text{case 1}:\alpha_i>0 \text{ means }y_i*(\vec w^Tz_i+b)=1,\text{ is the support vector}\\
&\text{case 2}:\alpha_i=0 \text{ is not the support vector}
\end{align}
$$

4. Sationary Condition: $\nabla_w L(w,\alpha,\beta)=0,\nabla_\beta L(w,\alpha,\beta)=0$

   > 因為當極值發生時，目標函式$f(w)$會與限制式$g_i(w),h_i(w)$相切，代表$\nabla_w f(w)=-\alpha_i \nabla_w g_i(w)$, $\nabla_w f(w)=-\beta_i \nabla_w h_i(w)$，又$L(w,\alpha,\beta)=f(w)+\sum_{i=1}^k\alpha_ig_i(w)+\sum_{i=1}^l\beta_ih_i(w)$，所以
   > $$
   > \nabla_w L(w,\alpha,\beta)=\frac{\partial L(w,\alpha,\beta)}{\partial w}=0
   > $$
   > 又希望可以符合等式限制式的條件
   > $$
   > \nabla_\beta L(w,\alpha,\beta)=\frac{\partial L(w,\alpha,\beta)}{\partial \beta}=0
   > $$
   > 至於不等式限制式的條件不一定會在邊界上，所以**不需要**$\nabla_\alpha L(w,\alpha,\beta)=\frac{\partial L(w,\alpha,\beta)}{\partial \alpha}=0$

$L(\vec w,b,\alpha)=\sum_{j=1}^m\frac{1}{2}w_j^2+\sum_{i=1}^n\alpha_i*[1-y_i*(\vec w^Tz_i+b)]]$

注意本例的$\vec w$與$b$皆是model parameter
$$
\frac{\partial L(\vec w,b,\alpha)}{\partial w}=0 
\Rightarrow \vec w=\sum_{i=1}^n\alpha_iy_iz_i\\
\frac{\partial L(\vec w,b,\alpha)}{\partial b}=0 
\Rightarrow \sum_{i=1}^n\alpha_iy_i=0\\
$$



得知$L(\vec w,b,\alpha)$的極大值發生在$\vec w=\sum_{i=1}^n\alpha_iy_iz_i$與$\sum_{i=1}^n\alpha_iy_i=0$之時，透過這個條件可以改寫$max_{\alpha}L(\vec w,b,\alpha)$
$$
max_{\alpha} \sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{k=1}^n\alpha_i\alpha_ky_iy_kz_i^Tz_k\\
s.t. \sum_{i=1}^n \alpha_i y_i=0, 0 \leq \alpha_i\\
$$
而kernel function $K(x_i,x_k)=\phi^T(x_i)\phi(x_k)$，所以上式可以改寫成

> Gram matrix $K=\Phi\Phi^T$
>
> where each element of the gram matrix is the kernel function $K(x_i,x_k)=\phi^T(x_i)\phi(x_k)$ 
>
> 矩陣的第i個row、第k個column都是一個kernel function $K(x_i,x_k)$，這個kernel function會吃進$x_i$、$x_j$兩個向量，因此由這個kernel function構成的矩陣滿足對稱與半陣定的性質

$$
max_{\alpha} \sum_{i=1}^n\alpha_i-\sum_{i=1}^n\sum_{k=1}^n\alpha_i\alpha_ky_iy_kK(x_i,x_k)\\
s.t. \sum_{i=1}^n \alpha_i y_i=0, 0 \leq \alpha_i\\
$$

至此我們可以透過quadratic programming來求出$\alpha_i$ (每筆資料都會對應到一個$\alpha_i$)

> Quadratic programming
> $$
> \min (1/2)x^TPx+q^Tx \\
> \begin{align}
> s.t. Gx &\leq h \\
> Ax&=b
> \end{align}
> $$

再來便可用$\alpha$來求出$\vec w$與$b$，不過觀察一下$\vec w=\sum_{i=1}^n\alpha_iy_iz_i$可以發現只有support vector($\alpha_i>0$) 才會對$\vec w$有貢獻，所以可以推得
$$
\Rightarrow
\left\{
\begin{aligned}
\vec w&=\sum_{i\in sv}\alpha_iy_iz_i\\
b&=y_{sv}-\sum_{i=1}^n\alpha_iy_iK(x_i,x_{sv})
\end{aligned}
\right.,
\text{sv means support vector}
$$

由此可知，我們可以直接用特徵轉換函式的內積，也就是kernel function來求出最佳的model parameter

不過同時我們也可以看出來SVM是個non-parametric的方法，所以雖然我們可以在高維去很好的切割出不同類別的data points，但很難找出 $\vec w$ 在原本的維度真正的樣子



題外話

通常梯度方向在圖上顯示的時候會是$(1,1,...,1,np.sum(\nabla f(w)))$，前面1的數量取決於w有幾個dimensions，也就是總共有多少個feature，因為想看的東西是當這些features都變動一點時，目標函式$f(w)$會成長多少
<!--<img src="./梯度下降法.mov" alt="梯度下降法" style="zoom:50%;" />-->



reference

> https://ithelp.ithome.com.tw/articles/10200121
>
> https://ithelp.ithome.com.tw/articles/10200301
>
> https://ithelp.ithome.com.tw/articles/10201279
>
> https://www.ycc.idv.tw/ml-course-techniques_2.html
>
> https://chih-sheng-huang821.medium.com/機器學習-支撐向量機-support-vector-machine-svm-詳細推導-c320098a3d2e
>
> https://zhuanlan.zhihu.com/p/32501517
>
> https://zhuanlan.zhihu.com/p/36503663
>
> https://ccjou.wordpress.com/2017/02/07/karush-kuhn-tucker-kkt-條件/
>
> https://www.cnblogs.com/gczr/p/10521551.html
>
> https://cvxopt.org/userguide/coneprog.html#quadratic-programming

