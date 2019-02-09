# -*- coding: utf-8 -*-
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def control_results(results):
    fig = plt.figure(figsize=(10,15))
    fig.suptitle("Control Results")
    
    gs = gridspec.GridSpec(3, 2, hspace=0.4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, :])
    
    ax1.plot(results['time'], results['x'], label='x')
    ax1.plot(results['time'], results['y'], label='y')
    ax1.plot(results['time'], results['z'], label='z')
    ax1.set_title('X-Y-Z')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(results['time'], results['x_velocity'], label='x_hat')
    ax2.plot(results['time'], results['y_velocity'], label='y_hat')
    ax2.plot(results['time'], results['z_velocity'], label='z_hat')
    ax2.set_title('X-Y-Z velocities')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(results['time'], results['phi'], label='phi')
    ax3.plot(results['time'], results['theta'], label='theta')
    ax3.plot(results['time'], results['psi'], label='psi')
    ax3.set_title('Phi-Theta-Psi')
    ax3.grid(True)
    ax3.legend()
    
    ax4.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    ax4.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    ax4.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    ax4.set_title('Phi-Theta-Psi Velocities')
    ax4.grid(True)
    ax4.legend()
    
    ax5.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    ax5.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    ax5.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    ax5.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    ax5.set_title('Rotor Revolutions')
    ax5.grid(True)
    ax5.legend()

def check_rewards(rewards, avg_rewards):
    fig = plt.figure(figsize=(7,5))
    fig.suptitle("Rewards")
    plt.plot(rewards, color = 'grey', alpha = 0.4)
    plt.plot(avg_rewards)
    plt.legend(['Total reward per episode', 'Average reward over last episodes'], fontsize = 'x-small')
