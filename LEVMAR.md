<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>

# 关于LEVMAR
## 1. 介绍
> levmar是一个Levenberg-Marquardt非线性优化算法(<http://www.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf>)库，基于GPL开源的的ANSI C实现。

## 2. 内容
> 包含constrained 和 unconstrained 两类优化函数

### unconstrained（非约束）函数：
|名称|精度|Jacobian形式|
|--|--|--|
|dlevmar_der()| double precision | analytic Jacobian.解析形式的jacobian矩阵|
|dlevmar_dif()| double precision | finite difference * approximated Jacobian.近似形式的jacobian矩阵|
|slevmar_der()| single precision | analytic Jacobian.解析形式的jacobian矩阵|
|slevmar_dif()| single precision | finite difference approximated Jacobian.近似形式的jacobian矩阵|

### unconstrained(带约束)函数：
|名称|精度|约束形式|Jacobian形式|
|--|--|--|--|
|dlevmar_lec_der()| double precision| linear equation constraints| analytic Jacobian
|dlevmar_lec_dif()| double precision| linear equation constraints| finite difference approximated Jacobian
|slevmar_lec_der()| single precision| linear equation constraints| analytic Jacobian
|slevmar_lec_dif()| single precision| linear equation constraints| finite difference approximated Jacobian
|dlevmar_bc_der()| double precision| box constraints| analytic Jacobian
|dlevmar_bc_dif()| double precision| box constraints| finite difference approximated Jacobian
|slevmar_bc_der()| single precision| box constraints| analytic Jacobian
|slevmar_bc_dif()| single precision| box constraints| finite difference approximated Jacobian
|dlevmar_blec_der()| double precision| box & linear equation constraints| analytic Jacobian
|dlevmar_blec_dif()| double precision| box & linear equation constraints| finite difference approximated |Jacobian
|slevmar_blec_der()| single precision| box & linear equation constraints| analytic Jacobian
|slevmar_blec_dif()| single precision| box & linear equation constraints| finite difference approximated Jacobian
|dlevmar_bleic_der()| double precision| box, linear equation & inequality constraints| analytic Jacobian
|dlevmar_bleic_dif()| double precision| box, linear equation & inequality constraints| finite difference approximated Jacobian
|slevmar_bleic_der()| single precision| box, linear equation & inequality constraints| analytic Jacobian
|slevmar_bleic_dif()| single precision| box, linear equation & inequality constraints| finite difference approximated Jacobian|

>Convenience wrappers xlevmar_blic_der()/xlevmar_blic_dif(),xlevmar_leic_der()/xlevmar_leic_dif() & xlevmar_lic_der()/xlevmar_lic_dif() to xlevmar_bleic_der()/xlevmar_bleic_dif() are also provided

* linear equation constraints: 线性等式约束
* box constraints：盒式约束，设定参数的上下限

## 3. 功能：
解决非线性最小二乘问题：


$$min\left\| x - f\left( p \right ) \right \|^{2},p\in R^{m},x\in R^{n}$$

$$f:R^{m} -> R^{n} ,m<=n$$

* box constraints：$lowerbound<=p\left [  i \right ]<=upperbound$
* linear equation constraints: $A_{k1\times m}p=b_{k1\times 1}$
## 4. 解释参数：
<h2><a name="Jac"></a><tt>dlevmar_der()</tt></h2>

```c++

/*
 * This function seeks the mx1 parameter vector p that best describes the nx1 measurements vector x.
 * 该函数求解一个mx1的参数向量p，使得参数f(p)趋近于nx1的目标向量x
 * All computations are double precision.双精度
 * An analytic Jacobian is required. In case the latter is unavailable or expensive to compute,
 * use dlevmar_dif() below.
 *
 * Returns the number of iterations (&gt;=0) if successful, -1 if failed
 *
 */

int dlevmar_der(
void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements.
                                                                 * $A p \in R^m -> a \hat{x} \in  R^n$
                                                                 */
void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
double *p,         /* I/O: initial parameter estimates. On output contains the estimated solution */
double *x,         /* I: measurement vector. NULL implies a zero vector */
int m,             /* I: parameter vector dimension (i.e. #unknowns) */
int n,             /* I: measurement vector dimension */
int itmax,         /* I: maximum number of iterations */
double opts[4],    /* I: minim. options [\tau, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                    * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                    */
double info[LM_INFO_SZ],
                    /* O: information regarding the minimization. Set to NULL if don't care
                     * info[0]= ||e||_2 at initial p.
                     * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, \mu/max[J^T J]_ii ], all computed at estimated p.
                     * info[5]= # iterations,
                     * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                     *                                 2 - stopped by small Dp
                     *                                 3 - stopped by itmax
                     *                                 4 - singular matrix. Restart from current p with increased \mu
                     *                                 5 - no further error reduction is possible. Restart with increased mu
                     *                                 6 - stopped by small ||e||_2
                     *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values; a user error
                     * info[7]= # function evaluations
                     * info[8]= # Jacobian evaluations
                     * info[9]= # linear systems solved, i.e. # attempts for reducing error
                     */
double *work,      /* I: pointer to working memory, allocated internally if NULL. If !=NULL, it is assumed to point to
                    * a memory chunk at least LM_DER_WORKSZ(m, n)*sizeof(double) bytes long
                    */
double *covar,     /* O: Covariance matrix corresponding to LS solution; Assumed to point to a mxm matrix.
                    * Set to NULL if not needed.
                    */
void *adata)       /* I: pointer to possibly needed additional data, passed uninterpreted to func &amp; jacf.
                    * Set to NULL if not needed
                    */
```
<h2><a name="Jac"></a><tt>dlevmar_dif()</tt></h2>

```c++
/*
 * Similar to dlevmar_der() except that the Jacobian is approximated internally with the aid of finite differences.
 *  和dlevmar_der()相似，该函数求解一个mx1的参数向量p，使得参数f(p)趋近于nx1的目标向量x.不同的是该函数内部使用一种近似的Jacobian方法。
 * Broyden's rank one updates are used to compute secant approximations to the Jacobian, effectively avoiding to call func several times for computing the finite difference approximations.
 * If the analytic Jacobian is available, use dlevmar_der() above.
 *
 * Returns the number of iterations if successful, -1 if failed
 *
 */
int dlevmar_dif(
void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements.主要的计算函数，计算误差
                                                                 * A p \in R^m yields a \hat{x} \in  R^n
                                                                 */
double *p,         /* I/O: initial parameter estimates. On output contains the estimated solution  参数指针，输入时要初始化值*/
double *x,         /* I: measurement vector. NULL implies a zero vector 目标向量，默认是0*/
int m,             /* I: parameter vector dimension (i.e. #unknowns) 参数向量维度*/
int n,             /* I: measurement vector dimension 目标向量维度*/
int itmax,         /* I: maximum number of iterations 最大迭代次数*/
double opts[5],    /* I: opts[0-4] = minim. options [\tau, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
                    * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and the
                    * step used in difference approximation to the Jacobian. If \delta<0, the Jacobian is approximated
                    * with central differences which are more accurate (but slower!) compared to the forward differences
                    * employed by default. Set to NULL for defaults to be used.
                    */
double info[LM_INFO_SZ],
                    /* O: information regarding the minimization. Set to NULL if don't care
                     * info[0]= ||e||_2 at initial p.
                     * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, \mu/max[J^T J]_ii ], all computed at estimated p.
                     * info[5]= # iterations,
                     * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                     *                                 2 - stopped by small Dp
                     *                                 3 - stopped by itmax
                     *                                 4 - singular matrix. Restart from current p with increased \mu
                     *                                 5 - no further error reduction is possible. Restart with increased mu
                     *                                 6 - stopped by small ||e||_2
                     *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values; a user error
                     * info[7]= # function evaluations
                     * info[8]= # Jacobian evaluations
                     * info[9]= # linear systems solved, i.e. # attempts for reducing error
                     */
double *work,      /* I: working memory, allocated internally if NULL. If !=NULL, it is assumed to point to
                    * a memory chunk at least LM_DIF_WORKSZ(m, n)*sizeof(double) bytes long
                    */
double *covar,     /* O: Covariance matrix corresponding to LS solution; Assumed to point to a mxm matrix.
                    * Set to NULL if not needed.
                    */
void *adata)       /* I: pointer to possibly needed additional data, passed uninterpreted to func.
                    * Set to NULL if not needed
                    */
```
# 安装：
<http://blog.sina.com.cn/s/blog_45b747f70101he1t.html>

levmar依赖clapack库，需要使用cmake生成解决方案\项目，然后编译生成库。

需要的资源有：

1. cmake-2.8.12.1-win32-x86.zip (选择Binary distributions栏下的第二个)。

2. clapack-3.2.1-CMAKE.tgz（页面做的稍乱，找到同名的那个压缩包下载）。

3. levmar-2.6

# 实例：

```c++
/* helical valley function, minimum at (1.0, 0.0, 0.0) */
#ifndef M_PI
#define M_PI   3.14159265358979323846  /* pi */
#endif
void helval(double *p, double *x, int m, int n, void *data)
{
double theta;

  if(p[0]<0.0)
     theta=atan(p[1]/p[0])/(2.0*M_PI) + 0.5;
  else if(0.0<p[0])
     theta=atan(p[1]/p[0])/(2.0*M_PI);
  else 
    theta=(p[1]>=0)? 0.25 : -0.25;

  x[0]=10.0*(p[2] - 10.0*theta);
  x[1]=10.0*(sqrt(p[0]*p[0] + p[1]*p[1]) - 1.0);
  x[2]=p[2];
}

void jachelval(double *p, double *jac, int m, int n, void *data)
{
register int i=0;
double tmp;

  tmp=p[0]*p[0] + p[1]*p[1];

  jac[i++]=50.0*p[1]/(M_PI*tmp);
  jac[i++]=-50.0*p[0]/(M_PI*tmp);
  jac[i++]=10.0;

  jac[i++]=10.0*p[0]/sqrt(tmp);
  jac[i++]=10.0*p[1]/sqrt(tmp);
  jac[i++]=0.0;

  jac[i++]=0.0;
  jac[i++]=0.0;
  jac[i++]=1.0;
}
```
```c++
void optim(){
    m=3; n=3;
    p[0]=-1.0; p[1]=0.0; p[2]=0.0;
    for(i=0; i<n; i++) x[i]=0.0;
    ret=dlevmar_der(helval, jachelval, p, x, m, n, 1000, opts, info, NULL, NULL, NULL); // with analytic Jacobian
    //ret=dlevmar_dif(helval, p, x, m, n, 1000, opts, info, NULL, NULL, NULL);  // no Jacobian
}

```
# Levenberg-Marquardt 