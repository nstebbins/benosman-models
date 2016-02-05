%% Benosman Neural Model

function [] = benosman()

    % generate voltage
    t = 0:1e-5:1.5; 
    
    V_syn = zeros(1, length(t)); % voltage synapses 
    ge_syn = zeros(1, length(t)); % g_e synapses
    gf_syn = zeros(1, length(t)); % g_f synapses
    gate_syn = zeros(1, length(t)); % gate synapses
    
    % put some synapses in 
    V_syn(1) = 9.5;
    ge_syn(1) = 36; 
    gf_syn(1) = 120; 
    gate_syn(1) = 0.99; 
    
    [V, g_f, t_k] = voltage(t, V_syn, ge_syn, gf_syn, gate_syn);

    % plot conductance
    figure; 
    subplot(2, 1, 1);
    plot(t, V);
    xlabel('t (sec)');
    ylabel('V');
    title('Output Voltage');
    
    % plot conductance
    subplot(2, 1, 2);
    plot(t, g_f);
    xlabel('t (sec)');
    ylabel('g_f');
    title('Conductance (g_{f})');
    
end

function [V, g_f, t_k] = voltage(t, V_syn, ge_syn, gf_syn, gate_syn)
    
    t = 1000*t; dt = diff(t(1:2));
    V = zeros(1, length(t)); V(1) = V_syn(1);
    g_f = zeros(1, length(t));
    t_k = [];
    
    V_t = 10; % threshold V (in mV)
    
    % conductances & gate variable (only g_f dynamic; values arbitrary)
    g_e = ge_syn(1); g_f(1) = gf_syn(1); gate = gate_syn(1);
    
    % time constants
    tau_f = 20; tau_m = 100 * 1000; % note: tau_m in seconds
    
    for i = 2:length(t), 
        % biases due to synapses
        gate = gate + gate_syn(i); g_e = g_e + ge_syn(i);
        % update conductance
        g_f(i) = g_f(i-1) + dt * ((-g_f(i-1)) / tau_f) + gf_syn(i);
        % update voltage (check for spike)
        V(i) = V(i-1) + dt * ((g_e + g_f(i) * gate) / tau_m) + V_syn(i);
        if(V(i) >= V_t), % reset
            V(i) = 0;
            g_e = 0; 
            g_f(i) = 0; 
            gate = 0;
            t_k = [t_k t(i)];
        end
    end
    
    t_k = t_k / 1000;

end