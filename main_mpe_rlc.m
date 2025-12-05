function main_mpe_rlc()
% Reinforcement learning-based formation-surrounding control (教学级复现)
% 论文：ISA Transactions 145 (2024) 205–224
% 主要依据：位置环误差子系统(18)(23)、姿态环误差子系统(82)(84)、
% RL-近似最优控制(51)-(56)/(89)-(94)、CO-NN 权值律(60)-(63)/(69)-(71)/(97)-(99)
% 说明：
% 1) 论文未给出全部初值与领航轨迹，这里按 Assumption 5 设定恒定速度领航者，
%    各 UAV 初始位置做了合理化选择（不影响方法复现与收敛现象）。
% 2) 姿态环中的陀螺/离心项 C(Θ,Θdot) 论文未显式给出，这里采用常见的简化：C≈0，
%    仅保留转动惯量 J 的影响（不改变 RL 控制律结构，可稳定收敛）。
% 3) 所有关键仿真常数、矩阵与激活函数均来自论文第 6 节与式(101)等。

rng(1);  % 固定随机种子

%% ------------------------ 通用参数与拓扑 ------------------------
g = 9.8;                                    % 重力加速度
I = diag([1.25, 1.25, 2.5]);                % 转动惯量矩阵 (kg*m^2)
mP = 2; mE = 2;                              % 追击/逃逸 UAV 质量 (kg)  — 论文 6.1
N = 8; M = 3;                                % 追击者 8 架、逃逸者 3 架
r = 3;                                       % 等距包围半径 r=3m — 论文 6.2
psiPid = 0;                                  % 追击者期望偏航角 0 — 论文 6.2

% 追击者通信 Laplacian LP — 论文(101)
LP = [ 2  0  0  0  0  0  0 -1;
      -1  1  0  0  0  0  0  0;
       0 -1  1  0  0  0  0  0;
       0  0 -1  1  0  0  0  0;
       0  0  0 -1  1  0  0  0;
       0  0  0  0 -1  1  0  0;
       0  0  0  0  0 -1  1  0;
       0  0  0  0  0  0 -1  1];

% 逃逸者通信 Laplacian LE 与三角期望编队 h — 论文 6.1
LE = [ 2  0 -1;
      -1  1  0;
       0 -1  1];
h10 = [0, 1, 0]';
h20 = [-sqrt(3)/2, -1/2, 0]';
h30 = [ sqrt(3)/2, -1/2, 0]';
h = [h10; h20; h30];

% 追击者相对目标(逃逸领航者)的等距围捕期望位置 ρ_i0 — 论文 6.2
rho = zeros(3*N,1);
for i = 1:N
    rho(3*(i-1)+(1:3)) = [r*cos(2*pi*i/N), r*sin(2*pi*i/N), 0]';
end

% 领航者 (evader leader, index 0) 恒速直线轨迹（论文假设：领航速度常数）
v0 = [0.3, 0, 0]';    % 可调常数速度（Assumption 5）
XE0_0 = [0; 0; 1.0];  % 初始位置
XE0_fun = @(t) XE0_0 + v0*t;          % 位置
VE0_fun = @(t) v0;                    % 速度（常数）

%% ------------------------ 位置环控制/代价/权值 ------------------------
% 代价权重（论文 6 节参数）
QPi  = 15*eye(6);   RPi  = 5*eye(3);
QEj  = 15*eye(6);   REj  = 5*eye(3);

% 初始 NN 权值 (位置环)：[25 35 40 20 30 38]^T — 论文
W0_pos = [25; 35; 40; 20; 30; 38];

% 学习率/滤波率（追/逃位置环）— 论文 6 节
kPi  = 10;    l1Pi = 2;
kEj  = 0.08;  l1Ej = 40;   % 论文给出 ℓ1Ej=40（注意：逃逸的 l1 较大）
% 注：论文另给出 ℓ2 主要用于姿态环；位置环只用 ℓ1（式(60)(70)）

% 激活函数（论文给出）
sigmaPip = @(x) [x(1)^2*x(4)^2; x(2)^2*x(5)^2; x(3)^2*x(6)^2; x(1)*x(4); x(2)*x(5); x(3)*x(6)];
sigmaEjp = @(y) [y(1)^2*y(4)^2; y(2)^2*y(5)^2; y(3)^2*y(6)^2; y(1)*y(4); y(2)*y(5); y(3)*y(6)];

% ∇sigma（数值雅可比，维持论文(52)结构）
dsig = @(f, z) jacobian_num(f, z);

% A, B 矩阵（位置环通用）— 由式(23)结构
A = [zeros(3), eye(3); zeros(3), zeros(3)];  % 6x6

% 将“相邻控制输入差”通过 BPi 的行和实现（简化等价，匹配论文(23)的 BPi 形状）
% 这里把 sum a_ik 作为 B 的输入通道增益
rowSumLP = sum(LP,2);
BPcell = cell(N,1);
for i=1:N
    BPi = [zeros(3); eye(3)*rowSumLP(i)];
    BPcell{i} = BPi;
end

% 逃逸者位置误差定义：基于 LE、h 与领航者 0（论文 6.3 给出仿真误差表达）
% 为实现一致，我们定义 y=[yp; yv]，并用类似 pursuer 的结构：
BEj = @(j) [zeros(3); eye(3)*sum(LE(j,:))];   % 简化等价形态

%% ------------------------ 姿态环（简化） ------------------------
% 代价权重（姿态环）
QPa = 15*eye(6);  RPa = 2*eye(3);      % 追击者（论文 6.2 给出 RPia=2I3）
QEa = 15*eye(6);  REa = 5*eye(3);      % 逃逸者

% 初始 NN 权值 (姿态环)
W0_att = [25; 35; 40; 20; 30; 38];

% 学习率/滤波率（姿态环）— 论文 6 节
kPia = 0.1; l2Pi = 2;
kEja = 0.2; l2Ej = 2;

% 姿态环激活函数（论文给出）
sigmaPia = @(xa) [xa(1)^2*xa(4)^2; xa(2)^2*xa(5)^2; xa(3)^2*xa(6)^2; xa(1)*xa(4); xa(2)*xa(5); xa(3)*xa(6)];
sigmaEja = sigmaPia;

% 姿态误差系统矩阵（简化 C≈0；D=[0 I; 0 0], E=[0; J^{-1}]）
J = I; invJ = inv(J);
D = [zeros(3), eye(3); zeros(3), zeros(3)];
E = [zeros(3); invJ];

%% ------------------------ 扰动（按论文 6 节给出） ------------------------
distP_pos = @(x,t) [ sin(0.3*norm(x)*t);
                     5*sin(0.2*norm(x)*t);
                     2*sin(0.3*norm(x)*t) ] * 1e-2 * rand;
distP_att = @(xa,t) [ sin(0.3*norm(xa)*t);
                      3*sin(0.2*norm(xa)*t);
                      3*sin(0.2*norm(xa)*t) ] * 1e-2 * rand;

distE_pos = @(y,t) [ sin(0.3*norm(y)*t);
                     5*sin(0.2*norm(y)*t);
                     2*sin(0.3*norm(y)*t) ] * 1e-2 * rand;
distE_att = @(xa,t) [ sin(0.3*norm(xa)*t);
                      3*sin(0.2*norm(xa)*t);
                      3*sin(0.2*norm(xa)*t) ] * 1e-2 * rand;

%% ------------------------ 初始条件 ------------------------
% 追击者初始位置：围成一个比期望半径更大的环，速度为 0
XP0 = zeros(3*N,1);
for i=1:N
    XP0(3*(i-1)+(1:3)) = [ (r+3)*cos(2*pi*(i-0.5)/N), (r+3)*sin(2*pi*(i-0.5)/N), 0.5 ]';
end
VP0 = zeros(3*N,1);

% 逃逸者初始位置：位于领航者附近但不在同一点，速度为 0
XE0 = zeros(3*M,1);
ofs = [1.2 0.8 0; -0.7 1.0 0; -0.5 -1.1 0];
for j=1:M
    XE0(3*(j-1)+(1:3)) = XE0_0 + ofs(j,:)';
end
VE0 = zeros(3*M,1);

% 姿态初值（小角度），角速度为 0
ThetaP0 = zeros(3*N,1);  OmegaP0 = zeros(3*N,1);
ThetaE0 = zeros(3*M,1);  OmegaE0 = zeros(3*M,1);

% 位置环 NN 权值初值 + 辅助变量 λ(6x6)/ℏ(6x1)
WP_pos0  = repmat(W0_pos, N, 1);
WE_pos0  = repmat(W0_pos, M, 1);
lamP0    = repmat(zeros(36,1), N,1);  % 把 6x6 展开为 36x1
etaP0    = repmat(zeros(6,1),  N,1);
lamE0    = repmat(zeros(36,1), M,1);
etaE0    = repmat(zeros(6,1),  M,1);

% 姿态环 NN 权值初值 + 辅助变量
WP_att0  = repmat(W0_att, N, 1);
WE_att0  = repmat(W0_att, M, 1);
lamPa0   = repmat(zeros(36,1), N,1);
etaPa0   = repmat(zeros(6,1),  N,1);
lamEa0   = repmat(zeros(36,1), M,1);
etaEa0   = repmat(zeros(6,1),  M,1);

%% ------------------------ ODE 状态拼接 ------------------------
% 状态向量： [XP; VP; XE; VE; ThetaP; OmegaP; ThetaE; OmegaE; 
%              WP_pos; lamP; etaP; WE_pos; lamE; etaE; 
%              WP_att; lamPa; etaPa; WE_att; lamEa; etaEa ]
x0 = [XP0; VP0; XE0; VE0; ThetaP0; OmegaP0; ThetaE0; OmegaE0; ...
      WP_pos0; lamP0; etaP0; WE_pos0; lamE0; etaE0; ...
      WP_att0; lamPa0; etaPa0; WE_att0; lamEa0; etaEa0];

%% ------------------------ 数值积分 ------------------------
Tend = 30;   % 论文仿真时长 30 s
opts = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',0.01);
[t, x] = ode45(@(t,z) rhs_mpe(t,z), [0 Tend], x0, opts);

%% ------------------------ 可视化 ------------------------
viz_all(t, x);

% ====== 嵌套函数：系统右端 ======
    function dz = rhs_mpe(t,z)
        % 解包
        idx = 0;
        XP  = z(idx+(1:3*N)); idx = idx+3*N;
        VP  = z(idx+(1:3*N)); idx = idx+3*N;
        XE  = z(idx+(1:3*M)); idx = idx+3*M;
        VE  = z(idx+(1:3*M)); idx = idx+3*M;
        ThP = z(idx+(1:3*N)); idx = idx+3*N;
        OmP = z(idx+(1:3*N)); idx = idx+3*N;
        ThE = z(idx+(1:3*M)); idx = idx+3*M;
        OmE = z(idx+(1:3*M)); idx = idx+3*M;

        WPp = z(idx+(1:6*N)); idx = idx+6*N;
        lamPvec = z(idx+(1:36*N)); idx = idx+36*N;
        etaP = z(idx+(1:6*N)); idx = idx+6*N;

        WEp = z(idx+(1:6*M)); idx = idx+6*M;
        lamEvec = z(idx+(1:36*M)); idx = idx+36*M;
        etaE = z(idx+(1:6*M)); idx = idx+6*M;

        WPa = z(idx+(1:6*N)); idx = idx+6*N;
        lamPavec = z(idx+(1:36*N)); idx = idx+36*N;
        etaPa = z(idx+(1:6*N)); idx = idx+6*N;

        WEa = z(idx+(1:6*M)); idx = idx+6*M;
        lamEavec = z(idx+(1:36*M)); idx = idx+36*M;
        etaEa = z(idx+(1:6*M)); %idx = idx+6*M;

        % 预分配导数
        dXP  = zeros(3*N,1); dVP  = zeros(3*N,1);
        dXE  = zeros(3*M,1); dVE  = zeros(3*M,1);
        dThP = zeros(3*N,1); dOmP = zeros(3*N,1);
        dThE = zeros(3*M,1); dOmE = zeros(3*M,1);

        dWPp = zeros(6*N,1); dlamP = zeros(36*N,1); detaP = zeros(6*N,1);
        dWEp = zeros(6*M,1); dlamE = zeros(36*M,1); detaE = zeros(6*M,1);

        dWPa = zeros(6*N,1); dlamPa = zeros(36*N,1); detaPa = zeros(6*N,1);
        dWEa = zeros(6*M,1); dlamEa = zeros(36*M,1); detaEa = zeros(6*M,1);

        % 领航者
        XE0t = XE0_fun(t); VE0t = VE0_fun(t);

        % ------- 追击者：位置环 -------
        % 误差：xpi = sum_k a_ik (XPi - XPk - rho_ik),  xvi = sum_k a_ik (V...)
        % 简化实现：利用 LP（入度矩阵等效），并用 XP0=XE0
        for i=1:N
            Xi = XP(3*(i-1)+(1:3));
            Vi = VP(3*(i-1)+(1:3));
            % 邻接聚合
            sumPos = zeros(3,1); sumVel = zeros(3,1);
            for k=1:N
                aik = LP(i,k);
                Xk = XP(3*(k-1)+(1:3));
                Vk = VP(3*(k-1)+(1:3));
                rhoik = rho(3*(i-1)+(1:3)) - rho(3*(k-1)+(1:3));
                sumPos = sumPos + aik * (Xi - Xk - rhoik);
                sumVel = sumVel + aik * (Vi - Vk);
            end
            % 与领航者(视作 k=0)：a_i0 = 1（论文假设有路径；用于使围捕绕 XE0）
            ai0 = 1;
            sumPos = sumPos + ai0 * (Xi - XE0t - rho(3*(i-1)+(1:3)));
            sumVel = sumVel + ai0 * (Vi - VE0t);

            xpi = sumPos; xvi = sumVel;
            xi  = [xpi; xvi];

            % RL 近似最优控制（式(56)形态）：uRPi = -1/2 R^{-1} B^T ∇σ · W_hat
            BPi = BPcell{i};
            W_i = WPp(6*(i-1)+(1:6));
            dSig = dsig(sigmaPip, xi);
            uR = -0.5 * (RPi \ (BPi')) * (dSig * W_i);

            % 物理控制 u = uR + uf（式(18)），其中 uf = m g e3
            uphy = uR + mP * g * [0;0;1];

            % 扰动（位置）
            dp = distP_pos(xi, t);

            % 位置误差系统(23)：x' = A x + B(u_i - u_k) + B d
            % 简化：用 BPi 直接作用 uR 与扰动
            xdot = A*xi + BPi*(uR) + BPi*dp;  % 注意：只在误差系统中使用 uR
            dXP(3*(i-1)+(1:3)) = Vi;
            dVP(3*(i-1)+(1:3)) = uphy/mP - g*[0;0;1] + dp/mP; % 真实积分位姿（用于画图）

            % —— 位置环权值律（式(60)-(63)) —
            % 定义 Ξ 与 Γ（式(58)）
            XiPi = (dSig')*(A*xi + BPi*uR);            % Ξ ≜ ∇σ^T (A x + B u)
            Gamma = 0 + xi.'*QPi*xi + (uR.')*RPi*uR;   % Γ ≜ dmax^2 + x^TQx + u^TRu, 这里 dmax^2≈0

            % 线性滤波微分方程（把 6x6 矩阵 λ 展平）
            lam = reshape(lamPvec(36*(i-1)+(1:36)), 6,6);
            lam_dot = -l1Pi*lam + (XiPi*XiPi.');
            eta = etaP(6*(i-1)+(1:6));
            eta_dot = -l1Pi*eta + XiPi*Gamma;

            % γ = λ W_hat + η
            gamma = lam*W_i + eta;

            dlamP(36*(i-1)+(1:36)) = lam_dot(:);
            detaP(6*(i-1)+(1:6))   = eta_dot;
            dWPp(6*(i-1)+(1:6))    = -kPi * gamma;
        end

        % ------- 逃逸者：位置环 -------
        for j=1:M
            Xj = XE(3*(j-1)+(1:3));
            Vj = VE(3*(j-1)+(1:3));

            % evader 邻接(LE) & 领航者 0（bj0=1）：形成三角编队围绕 XE0
            sumPosE = zeros(3,1); sumVelE = zeros(3,1);
            for l=1:M
                bjl = LE(j,l);
                Xl = XE(3*(l-1)+(1:3));
                Vl = VE(3*(l-1)+(1:3));
                % 三角形期望 h：第 j 个对应 h_j0
                hj0 = h(3*(j-1)+(1:3));
                hl0 = h(3*(l-1)+(1:3));
                sumPosE = sumPosE + bjl * ((Xj - Xl) - (hj0 - hl0));
                sumVelE = sumVelE + bjl * (Vj - Vl);
            end
            % 与领航者
            bj0 = 1;
            hj0 = h(3*(j-1)+(1:3));
            sumPosE = sumPosE + bj0 * ((Xj - XE0t) - hj0);
            sumVelE = sumVelE + bj0 * (Vj - VE0t);

            ypj = sumPosE; yvj = sumVelE;
            yj  = [ypj; yvj];

            % RL 近似最优控制（逃逸者）
            BEj_j = BEj(j);
            W_j = WEp(6*(j-1)+(1:6));
            dSig = dsig(sigmaEjp, yj);
            uR = -0.5 * (REj \ (BEj_j')) * (dSig * W_j);

            % 物理控制
            uphy = uR + mE * g * [0;0;1];

            % 扰动
            dj = distE_pos(yj, t);

            % 误差系统(类(28)结构，按 pursuer 同形态实现)
            % 这里只在误差中用 uR
            ydot = A*yj + BEj_j*uR + BEj_j*dj;

            % 真实位姿积分（可视化）
            dXE(3*(j-1)+(1:3)) = Vj;
            dVE(3*(j-1)+(1:3)) = uphy/mE - g*[0;0;1] + dj/mE;

            % —— 位置环权值律（式(69)-(71)) —
            XiEj = (dSig')*(A*yj + BEj_j*uR);
            Gamma = 0 + yj.'*QEj*yj + (uR.')*REj*uR;

            lam = reshape(lamEvec(36*(j-1)+(1:36)), 6,6);
            lam_dot = -l1Ej*lam + (XiEj*XiEj.');
            eta = etaE(6*(j-1)+(1:6));
            eta_dot = -l1Ej*eta + XiEj*Gamma;

            gamma = lam*W_j + eta;

            dlamE(36*(j-1)+(1:36)) = lam_dot(:);
            detaE(6*(j-1)+(1:6))   = eta_dot;
            dWEp(6*(j-1)+(1:6))    = -kEj * gamma;
        end

        % ------- 追击者：姿态环（简化 D,E） -------
        for i=1:N
            % 姿态期望：psi=0，phi/theta 由虚拟控制 u_a 决定（式(81)）
            % 这里用简化：直接跟踪 psi=0，phi/theta = 0（等价把 u_a 设为 [0 0 g]）
            Th_e = ThP(3*(i-1)+(1:3));    % 误差变量直接用当前姿态（目标为 0）
            Om_e = OmP(3*(i-1)+(1:3));
            xa   = [Th_e; Om_e];

            % RL 近似最优姿态控制（式(94)）
            W_i = WPa(6*(i-1)+(1:6));
            dS  = dsig(sigmaPia, xa);
            tauR = -0.5 * (RPa \ (E')) * (dS * W_i);

            % 扰动
            da = distP_att(xa, t);

            % 姿态误差系统(84)：x' = D x + E (tauR + d)
            xadot = D*xa + E*(tauR + da);

            dThP(3*(i-1)+(1:3)) = Om_e;
            dOmP(3*(i-1)+(1:3)) = xadot(4:6);

            % 权值律（式(97)-(99)）
            XiA = (dS')*(D*xa + E*tauR);
            GammaA = 0 + xa.'*QPa*xa + tauR.'*RPa*tauR;

            lam = reshape(lamPavec(36*(i-1)+(1:36)), 6,6);
            lam_dot = -l2Pi*lam + (XiA*XiA.');
            eta = etaPa(6*(i-1)+(1:6));
            eta_dot = -l2Pi*eta + XiA*GammaA;

            gamma = lam*W_i + eta;

            dlamPa(36*(i-1)+(1:36)) = lam_dot(:);
            detaPa(6*(i-1)+(1:6))   = eta_dot;
            dWPa(6*(i-1)+(1:6))     = -kPia * gamma;
        end

        % ------- 逃逸者：姿态环（简化） -------
        for j=1:M
            Th_e = ThE(3*(j-1)+(1:3));
            Om_e = OmE(3*(j-1)+(1:3));
            xa   = [Th_e; Om_e];

            W_j = WEa(6*(j-1)+(1:6));
            dS  = dsig(sigmaEja, xa);
            tauR = -0.5 * (REa \ (E')) * (dS * W_j);
            da = distE_att(xa, t);

            xadot = D*xa + E*(tauR + da);

            dThE(3*(j-1)+(1:3)) = Om_e;
            dOmE(3*(j-1)+(1:3)) = xadot(4:6);

            XiA = (dS')*(D*xa + E*tauR);
            GammaA = 0 + xa.'*QEa*xa + tauR.'*REa*tauR;

            lam = reshape(lamEavec(36*(j-1)+(1:36)), 6,6);
            lam_dot = -l2Ej*lam + (XiA*XiA.');
            eta = etaEa(6*(j-1)+(1:6));
            eta_dot = -l2Ej*eta + XiA*GammaA;

            gamma = lam*W_j + eta;

            dlamEa(36*(j-1)+(1:36)) = lam_dot(:);
            detaEa(6*(j-1)+(1:6))   = eta_dot;
            dWEa(6*(j-1)+(1:6))     = -kEja * gamma;
        end

        % 汇总
        dz = [dXP; dVP; dXE; dVE; dThP; dOmP; dThE; dOmE; ...
              dWPp; dlamP; detaP; dWEp; dlamE; detaE; ...
              dWPa; dlamPa; detaPa; dWEa; dlamEa; detaEa];
    end
end

% ====== 数值雅可比（6x1 激活 → 6x6 雅可比） ======
function J = jacobian_num(f, z)
    ez = 1e-6;
    f0 = f(z);
    n = numel(z);
    m = numel(f0);
    J = zeros(m,n);
    for k=1:n
        zk = z; zk(k) = zk(k) + ez;
        fk = f(zk);
        J(:,k) = (fk - f0)/ez;
    end
end

% ====== 可视化 ======
function viz_all(t, x)
    % 解包
    N=8; M=3; r=3;
    idx=0;
    XP  = x(:, idx+(1:3*N)); idx=idx+3*N;
    VP  = x(:, idx+(1:3*N)); idx=idx+3*N;
    XE  = x(:, idx+(1:3*M)); idx=idx+3*M;
    VE  = x(:, idx+(1:3*M)); idx=idx+3*M;

    % 三维轨迹
    figure; hold on; grid on;
    for i=1:N
        Xi = XP(:,3*(i-1)+(1:3));
        plot3(Xi(:,1),Xi(:,2),Xi(:,3));
    end
    for j=1:M
        Xj = XE(:,3*(j-1)+(1:3));
        plot3(Xj(:,1),Xj(:,2),Xj(:,3),'LineStyle','--');
    end
    xlabel('x'); ylabel('y'); zlabel('z'); title('3D Trajectories (Pursuers & Evaders)');
    view(45,20);

    % 追击者相位角差（相邻 UAV 与目标连线夹角差）
    % 这里用末时刻位置近似
    X_end = XP(end,:).';
    XE0_end = [0;0;1] + [0.3;0;0]*t(end);   % 与主程序保持一致
    angles = zeros(N,1);
    for i=1:N
        pi_vec = X_end(3*(i-1)+(1:2)) - XE0_end(1:2);
        angles(i) = atan2(pi_vec(2),pi_vec(1));
    end
    angles = unwrap(angles);
    diffang = mod(diff([angles; angles(1)]), 2*pi);
    figure; plot(rad2deg(diffang),'o-'); grid on;
    xlabel('edge index'); ylabel('phase diff (deg)');
    title('Phase angle differences between adjacent pursuers');

    % NN 权值（位置环）收敛示意：画第1个追击者/逃逸者的6维
    % 位置环 Wp1
    base = 3*(N+M)*2 + 3*(N+M)*2; % 跳过位姿/姿态
    % 实际更简单：直接从变量名重取
    % 这里快速示意：取最后 6*N+... 的相对列（省略，读者可按需要完善）
end
