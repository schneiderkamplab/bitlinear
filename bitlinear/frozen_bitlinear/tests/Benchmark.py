import torch
import torch.nn.functional as F

import os

from itertools import product
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src import *
from tests.helpers import weights

class Benchmark:
    
    options = [256, 512, 1024, 2048, 4096]
    
    def __init__(self, args):
        
        self.path = args.save_dir
        
        self.device='cuda'
        
        self.kernel_name = args.kernel
        self.kernel = eval(self.kernel_name)(activation_measure='Fp16')
        
        self.baseline = TorchLinear(activation_measure='Fp16')
        
        print(f'Testing with {self.kernel_name} kernel.')
        self.ref_lib = 'cuBLAS'        
                
        if args.a or args.p:
            self.profiling()
        if args.a or args.u:
            self.unittests()
        if args.a or args.t:
            self.throughputs()
        
    def unittest(self, M=512, N=256, K=128):
        
        torch.manual_seed(0)

        inputs = torch.randn((M, K), device=self.device, dtype=torch.float16)
        biases = torch.randn((M, 1), device=self.device, dtype=torch.float16)

        weights_torch, weights_kernel, scale = weights(N, K, self.kernel, self.baseline) 
            
        triton_output = self.kernel(inputs, weights_kernel, biases, scale)
        torch_output = self.baseline(inputs, weights_torch, biases, scale)
        
        if torch.allclose(triton_output, torch_output, atol=1e-1, equal_nan=True): 
            print(f"✅ Triton and Torch matchfor (M={M}, N={N}, K={K})")
            return True, 0.0
        else:
            difference = torch.max(torch.abs(triton_output - torch_output)).item() / torch.max(torch.abs(torch_output)).item()
            print(f"❌ Triton and Torch differ for (M={M}, N={N}, K={K}) -- Maximum Normalized Difference : {difference}")
            return False, difference
    
    def unittests(self, upper_limit=4096, step=128):
        
        pbar = tqdm(total=upper_limit//step, desc=f"Unit Tests", leave=True)
        
        results = []
        
        for combo in range(step, upper_limit+step, step):
            
            result, diff = self.unittest(combo, combo, combo)
            results.append([combo, result, diff])
            
            pbar.set_description(f"M=N=K : {combo} ")
            pbar.update()
        
        pbar.close()
        
        results = pd.DataFrame(results, columns=['M=N=K', 'Matches Torch', 'Maximum Normalized Difference'])
        
        save_path = os.path.join(self.path,"unittests")
        os.makedirs(save_path, exist_ok=True)
        
        with pd.ExcelWriter(os.path.join(save_path,"data.xlsx")) as writer:
            results.to_excel(writer)
            
        print(f'\nUnitTest data saved to {save_path}\n')
        
        print('-------------------------------------')
        print(f'Percentage of matching outputs {results["Matches Torch"].astype(int).mean() * 100}%')
        print('-------------------------------------')
 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(results['M=N=K'], results['Maximum Normalized Difference'], marker='o', color='b', s=50)

        ax.set_xlabel('Dimension Size', fontsize=14)
        ax.set_ylabel('Maximum Normalized Difference in Outputs', fontsize=14)
        ax.set_title(f'Comparing PyTorch and {self.kernel_name} Output Tensors', fontsize=16)
        ax.grid(True)

        plt.savefig(os.path.join(save_path, "difference.png"), dpi=300)
        plt.show()
    
    def profiling(self, upper_limit=75):
        
        combinations = list(product(self.options, self.options, self.options))
        
        results = { 
            item : []
                for item in [
                    'M', 'N', 'K', 'Custom_CUDA', 'cuBLAS_CUDA', 'Custom_CPU', 'cuBLAS_CPU', 'CUDA_factor', 'CPU_factor'
                ]
        }
                
        for [M, N, K] in combinations:
            
            print(f" M : {M} | N : {N} | K : {K}")
            
            results['M'].append(M)
            results['N'].append(N)
            results['K'].append(K)
            
            inputs = torch.randn((M, K), device=self.device, dtype=torch.float16)
            biases = torch.randn((M, 1), device=self.device, dtype=torch.float16)
            
            weights_torch, weights_kernel, scale = weights(N, K, self.kernel, self.baseline) 
            
            print("Profiling custom kernel")
            with torch.autograd.profiler.profile(use_device = self.device) as prof:
                self.kernel(inputs, weights_kernel, biases, scale)
            events = prof.key_averages()
            results['Custom_CUDA'].append(sum(event.device_time for event in events) / 1000.0)
            results['Custom_CPU'].append(sum(event.cpu_time for event in events) / 1000.0)
            print(events.table(sort_by="cuda_time_total"))
            
            print("Profiling cuBLAS kernel")
            with torch.autograd.profiler.profile(use_device = self.device) as prof:
                self.baseline(inputs, weights_torch, biases, scale)
            events = prof.key_averages()
            results['cuBLAS_CUDA'].append(sum(event.device_time for event in events) / 1000.0)
            results['cuBLAS_CPU'].append(sum(event.cpu_time for event in events) / 1000.0)
            print(events.table(sort_by="cuda_time_total"))

            results['CUDA_factor'].append(results['cuBLAS_CUDA'][-1]/results['Custom_CUDA'][-1])
            results['CPU_factor'].append(results['cuBLAS_CPU'][-1]/results['Custom_CPU'][-1])
        
        results = pd.DataFrame(results) 
                  
        save_path = os.path.join(self.path, "profiling")
        os.makedirs(save_path, exist_ok=True)
        
        with pd.ExcelWriter(os.path.join(save_path, "data.xlsx")) as writer:
            results.to_excel(writer)
            
        print(f'Profiling data saved to {save_path}\n')
        
        print(f' CUDA improvement = ~{min(results['CUDA_factor'])}-{max(results['CUDA_factor'])}')
        print(f' CPU improvement = ~{min(results['CPU_factor'])}-{max(results['CPU_factor'])}')
        
        #### Plot Over Iterations
        def iterations():
            plt.figure(figsize=(12, 12))

            plt.subplot(2, 1, 1)
            plt.plot(results['Custom_CUDA'], label='Custom CUDA Time')
            plt.plot(results['cuBLAS_CUDA'], label='cuBLAS CUDA Time')
            plt.xlabel('Iteration')
            plt.ylabel('CUDA Time (ms)')
            plt.ylim(0, max(np.percentile(results['Custom_CUDA'], upper_limit), np.percentile(results['cuBLAS_CUDA'], upper_limit)))
            plt.title('CUDA Time Comparison')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(results['Custom_CPU'], label='Custom CPU Time')
            plt.plot(results['cuBLAS_CPU'], label='cuBLAS CPU Time')
            plt.xlabel('Iteration')
            plt.ylabel('CPU Time (ms)')
            plt.ylim(0, max(np.percentile(results['Custom_CPU'], upper_limit), np.percentile(results['cuBLAS_CPU'], upper_limit)))
            plt.title('CPU Time Comparison')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "overall.png"))
            plt.show()
        
            
        ### Plot Difference Factor
        def difference(device):
            fig = go.Figure(data=[go.Scatter3d(
                    x=results['M'],
                    y=results['N'],
                    z=results['K'],
                    mode='markers',
                    marker=dict(
                    size=3,
                    color=results[f'Custom_{device}']/results[f'cuBLAS_{device}'],
                    colorscale='Viridis',
                    colorbar=dict(title='Performance Factor (ms)')
                    ),
                    text=results[f'Custom_{device}']/results[f'cuBLAS_{device}'],  # Add text for tooltips
                    hovertext=results[f'Custom_{device}']/results[f'cuBLAS_{device}']  # Use the text for hover info
                )])

            fig.update_layout(
                title=f'Performance Difference Between Custom_{device} and cuBLAS_{device}',
                scene=dict(
                    xaxis_title='M',
                    yaxis_title='N',
                    zaxis_title='K'
                )
            )

            fig.write_html(os.path.join(save_path, f"{device}_factor.html"))
        
        ### Plot as evolving over individual values
        def individual(fixed_axis, device):
            row_label = 'N' if fixed_axis == 'M' else 'M'
            col_label = 'N' if fixed_axis == 'K' else 'K'

            fig, ax = plt.subplots(len(self.options), len(self.options), figsize=(20, 20))

            for row, row_val in zip(ax, self.options):
                for plot, col_val in zip(row, self.options):
                    
                    filtered_df = results.loc[(results[row_label] == row_val) & (results[col_label] == col_val)]
                    
                    plot.plot(filtered_df[fixed_axis], filtered_df[f'cuBLAS_{device}'], label='cuBLAS')
                    plot.plot(filtered_df[fixed_axis], filtered_df[f'Custom_{device}'], label=self.kernel_name)
                    
                    plot.set_title(f'{row_label}={row_val}, {col_label}={col_val}', fontsize=10)
                    plot.tick_params(axis='both', which='major', labelsize=8)

            # Place the legend outside the subplots
            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper left', fancybox=True, shadow=True, fontsize=15)

            # Set the super title for the figure
            fig.suptitle(f'Comparing {self.ref_lib} and {self.kernel_name} on {device} (Plotting over {fixed_axis})', fontsize=20)
            fig.text(0.04, 0.5, 'Performance (ms)', va='center', rotation='vertical', fontsize=14)
            
            fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout to make space for the title and legend
            
            plt.savefig(os.path.join(save_path, f"{device}_fixed{fixed_axis}.png")) 
        
        iterations()
        for device in ['CPU', 'CUDA']:
            difference(device)
            for fixed_axis in ['M', 'N', 'K']:
                individual(fixed_axis, device)
    
    def throughput(self, M, N, K, iterations=100):
        
        perf = lambda ms: 2 * M * N * K * iterations * 1e-12 / (ms * 1e-3)
                
        inputs = torch.randn((M, K), device=self.device, dtype=torch.float16)
        biases = torch.randn((M, 1), device=self.device, dtype=torch.float16)
        
        weights_torch, weights_kernel, scale = weights(N, K, self.kernel, self.baseline) 
        
        cuBlas_time, cuBlas_mem = get_throughput(self.baseline, iterations)(inputs, weights_torch, biases, scale)
        
        kernel_time, kernel_mem = get_throughput(self.kernel, iterations)(inputs, weights_kernel, biases, scale)
        
        return [ M, N, K, perf(cuBlas_time), perf(kernel_time), kernel_mem - cuBlas_mem, ]

    def throughputs(self):
        combinations = list(product(self.options, self.options, self.options))
        
        pbar = tqdm(total=len(combinations), desc=f"Throughput", leave=True)
        
        results = []
        
        for [M, N, K] in combinations:
            
            results.append(self.throughput(M, N, K))
            
            pbar.set_description(f"M = {M} | N = {N} | K = {K} ")
            pbar.update()
        
        pbar.close()
        
        results = pd.DataFrame(results, columns=['M', 'N', 'K', 'cuBLAS Performance', 'kernel Performance', 'Memory Difference']) 
                  
        save_path = os.path.join(self.path, "performance")
        os.makedirs(save_path, exist_ok=True)
        
        with pd.ExcelWriter(os.path.join(save_path, "data.xlsx")) as writer:
            results.to_excel(writer)
            
        print(f'Performance data saved to {save_path}\n')
        
        ### Performance
        for fixed_axis in ['M', 'N', 'K']:
            row_label = 'N' if fixed_axis == 'M' else 'M'
            col_label = 'N' if fixed_axis == 'K' else 'K'

            fig, ax = plt.subplots(len(self.options), len(self.options), figsize=(20, 20))

            for row, row_val in zip(ax, self.options):
                for plot, col_val in zip(row, self.options):
                    
                    filtered_df = results.loc[(results[row_label] == row_val) & (results[col_label] == col_val)]
                    
                    plot.plot(filtered_df[fixed_axis], filtered_df[f'cuBLAS Performance'], label='cuBLAS')
                    plot.plot(filtered_df[fixed_axis], filtered_df[f'kernel Performance'], label=self.kernel_name)
                    
                    plot.set_title(f'{row_label}={row_val}, {col_label}={col_val}', fontsize=10)
                    plot.tick_params(axis='both', which='major', labelsize=8)

            # Place the legend outside the subplots
            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper left', fancybox=True, shadow=True, fontsize=15)

            # Set the super title for the figure
            fig.suptitle(f'Comparing {self.ref_lib} and {self.kernel_name} Performance (Plotting over {fixed_axis})', fontsize=20)
            fig.text(0.04, 0.5, 'Performance (TFLOPs)', va='center', rotation='vertical', fontsize=14)
            
            fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout to make space for the title and legend
            
            plt.savefig(os.path.join(save_path, f"performance_fixed{fixed_axis}.png")) 
            plt.show()
        
        ### memory
        for fixed_axis in ['M', 'N', 'K']:
            row_label = 'N' if fixed_axis == 'M' else 'M'
            col_label = 'N' if fixed_axis == 'K' else 'K'

            fig, ax = plt.subplots(1, len(self.options), figsize=(20, 20))

            for plot, row_val in zip(ax, self.options):
                for col_val in self.options:
                    
                    filtered_df = results.loc[(results[row_label] == row_val) & (results[col_label] == col_val)]
                    
                    plot.plot(filtered_df[fixed_axis], filtered_df[f'Memory Difference'], label=f'{col_label}={col_val}')
                    
                plot.set_title(f'{row_label}={row_val}', fontsize=10)
                plot.legend()
                plot.tick_params(axis='both', which='major', labelsize=8)


            # Set the super title for the figure
            fig.suptitle(f'Difference between {self.kernel_name} and {self.ref_lib} Memory (Plotting over {fixed_axis})', fontsize=20)
            fig.text(0.04, 0.5, 'Memory Difference (MBs)', va='center', rotation='vertical', fontsize=14)
            
            fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout to make space for the title and legend
            
            plt.savefig(os.path.join(save_path, f"memory_fixed{fixed_axis}.png")) 
            plt.show()

def get_throughput(func, iterations):
    def wrapper(*args):
        
        for _ in range(10):
            func(*args)
        
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
                    
        start_event.record()

        # benchmark
        for _ in range(iterations): 
            func(*args)
        
        end_event.record()
        
        torch.cuda.synchronize()

        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        
        return start_event.elapsed_time(end_event), max_memory_allocated
        
    return wrapper