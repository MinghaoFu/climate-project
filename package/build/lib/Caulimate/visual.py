import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ipywidgets as widgets

from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from tqdm import tqdm
from lingam.utils import make_dot
import subprocess

from .tools import makedir, check_array
from .graph import bin_mat, threshold_till_dag, is_dag

def fig_to_image(fig):
    """
    Convert a matplotlib figure to an RGB image.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.
    
    Returns
    -------
    np.array
        An RGB image.
    """
    fig.canvas.draw() # Draw the figure onto the canvas
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # Get the RGB buffer
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Reshape the buffer to an image
    return img

def video_from_figs_list(figs_list, video_save_path):
    """
    Create video from a list of matplotlib figures.
    
    Parameters
    ----------
    figs_list : list
        List of matplotlib figures.
    video_path : str
        Path where the video will be saved.
    """
    frames = [] # List to store frames
    #print('--- Create video from {} figures'.format(len(figs_list)))
    for fig in tqdm(figs_list, desc="Processing Figures to Video"):
        # Convert each figure to an image and append to frames
        img = fig_to_image(fig)
        frames.append(img)
    
    # Save frames as a video
    imageio.mimsave(video_save_path, frames, fps=10)

def video_from_figs_dir(figs_dir, video_save_path, suffix='.png'):  
    """
    Create video from index-ordering figures.
    
    Parameters
    ----------
    figs_dir : 
        Directory storing figures.
    video_path :
        Video saved path.
    """
    img_paths = sorted(glob.glob(f'{figs_dir}/*{suffix}'))[:10]
    frames = []
    #print('--- Create video from {} figures'.format(len(img_paths)))
    for path in tqdm(img_paths):
        frames.append(imageio.imread(path))

    imageio.mimsave(video_save_path, frames, fps=10)  
    
    
def plot_solutions(mats, names, save_path=None, add_value=False):
    """Checkpointing after the training ends.

    Args:
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
        save_name (str or None): Filename to solve the plot. Set to None
            to disable. Default: None.
    """
    # Define a function to add values to the plot
    def add_values_to_plot(ax, matrix):
        for (i, j), val in np.ndenumerate(matrix):
            if np.abs(val) > 0.1:
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')
    
    n_figs = len(mats)
    if n_figs > 1:
        fig, axes = plt.subplots(figsize=(10, n_figs), ncols=n_figs)

        for i in range(n_figs):
            im = axes[i].imshow(mats[i], cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
            axes[i].set_title(names[i], fontsize=13)
            if i != 0:
                axes[i].set_yticklabels([])    # Remove yticks
            axes[i].tick_params(labelsize=13)
            if add_value:
                add_values_to_plot(axes[i], mats[i])
            
        # Adjust space between subplots and add colorbar
        fig.subplots_adjust(wspace=0.1)
        im_ratio = len(mats) / 10
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
        cbar.ax.tick_params(labelsize=13)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

        # Plot the single matrix
        im = ax.imshow(mats[0], cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
        ax.set_title(names[0], fontsize=13)
        ax.tick_params(labelsize=13)

        # Add values to the plot if required
        if add_value:
            add_values_to_plot(ax, mats[0])

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.035)
        cbar.ax.tick_params(labelsize=13)
        
    # Save or display the figure
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
    return fig

def save_DAG(n_plots, save_dir, epoch, Bs_pred, Bs_gt=None, graph_thres=0.1, add_value=True, ):
    print(f'--- Save DAG predictions...')
    save_epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    makedir(save_epoch_dir)
    
    Bs_pred = check_array(Bs_pred) 
    Bs_pred[:, np.mean(np.abs(Bs_pred), axis=0) < graph_thres] = 0
    if Bs_gt is None:
        Bs_gt = Bs_pred
    else:
        Bs_gt = check_array(Bs_gt)
        
    row_indices, col_indices = np.nonzero(Bs_gt[0])
    edge_values = Bs_gt[:, row_indices, col_indices]
    values = []
    values_true =[]
    for _ in range(len(edge_values)):
        values.append([])
        values_true.append([])
    for k in range(n_plots):
        for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
            values[idx].append(Bs_pred[k][i, j])
            values_true[idx].append(Bs_gt[k][i, j])
    
    for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
        plt.plot(values[idx], label='Pred' + str(idx))
        plt.plot(values_true[idx], label = 'True' + str(idx))
        plt.legend() 
        plt.savefig(os.path.join(save_epoch_dir, f'({i}, {j})_trend.png'), format='png')
        plt.show()
        plt.clf()
        
    time_idx = np.random.randint(0, n_plots) # plot one time index
    plot_solutions([Bs_gt[time_idx], Bs_pred[time_idx]], ['B_gt', 'B_est'], os.path.join(save_epoch_dir, f'DAG_{time_idx}.png'), add_value=add_value)
    np.save(os.path.join(save_epoch_dir, 'prediction.npy'), np.round(Bs_pred, 4))
    np.save(os.path.join(save_epoch_dir, 'ground_truth.npy'), np.round(Bs_gt, 4))
    
def make_dots(arr: np.array, labels, save_path, name):
    if len(arr.shape) > 2:
        for i in arr.shape[0]:
            dot = make_dot(arr[i])
            dot.format = 'png'
            dot.render(os.path.join(save_path, f'{name}_{i}'))
    elif len(arr.shape) == 2:
        dot = make_dot(arr, labels=labels)
        dot.format = 'png'
        dot.render(os.path.join(save_path, name))
        os.remove(os.path.join(save_path, name)) # remove digraph
        
def plot_adj_mat_on_map(adj_mat, coords, fig=None, ax=None, extent=None, plot_dot=True, save_path=None, threshold=0.1):
    """plot adjacency matrix on Earth map

    Args:
        adj_mat (_type_): _description_
        coords (_type_): _description_
        ax (_type_, optional): _description_. Defaults to None.
        extent (_type_, optional): Earth extent. Defaults to None.
        plot_dot (bool, optional): plot the physical coordination as dots or not. Defaults to True.
        save_path (_type_, optional): path to save image. Defaults to None.
        threshold (float, optional): edge of threshold. Defaults to 0.1.

    Returns:
        _type_: _description_
    """

    adj_mat = check_array(adj_mat)
    coords = check_array(coords)[:, ::-1]
    
    fig = plt.figure(figsize=(10, 6))
    projection = ccrs.PlateCarree(central_longitude=(lonW + lonE) / 2)
    if ax is None:
        
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        
        # Add features to the map
        # ax.add_feature(cfeature.LAND, color='lightgreen')   # Color the land
        # ax.add_feature(cfeature.OCEAN, color='lightblue')   # Color the oceans
        ax.add_feature(cfeature.COASTLINE)                  # Add coastlines
        ax.add_feature(cfeature.BORDERS, linestyle=':')     # Add country borders
        # ax.add_feature(cfeature.LAKES, color='lightblue')   # Color the lakes
        # ax.add_feature(cfeature.RIVERS)                     # Add rivers
        # Define the geographic bounds of the map
        #ax.set_extent([np.min(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 0]), np.max(coords[:, 1])])  # Adjust as needed
        if extent is not None:
            ax.set_extent(extent, crs=projection)  # Adjust as needed
        #extent = [lonW - (lonW + lonE)/2, lonE - (lonW + lonE)/2, latS, latN]
        
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if np.abs(adj_mat[i, j]) > threshold:
                start_coords = coords[j]
                end_coords = coords[i]
                # Plot the start and end points
                if plot_dot:
                    ax.plot(start_coords[0], start_coords[1], color='white', marker='o', markersize=1, transform=ccrs.Geodetic())
                    ax.plot(end_coords[0], end_coords[1], color='white', marker='o', markersize=1, transform=ccrs.Geodetic())

                # Draw a customized arrow from start to end point
                ax.annotate('', xy=(end_coords[0], end_coords[1]), xytext=(start_coords[0], start_coords[1]),
                            arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.4", color='red', lw=0.1),
                            transform=ccrs.Geodetic())
    
    if save_path is not None:      
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig, ax


def figures_slider_display(fig_list):
    """
    fig_list: list of plt.figure() objects
    save_path: save path
    """
    def on_slider_change(change):
        index = change['new']
        
        fig = fig_list[index]
        with out:
            clear_output(wait=True)
            display(fig)
    
    max_len = len(fig_list)
    # Create a slider widget with a range large enough for your figures
    slider = widgets.IntSlider(
        min=0, max=max_len,  # Set the max value to the number of figures you have
        step=1, value=0,
        description='Figure Index:',
        continuous_update=False  # Update only when the slider is released
    )

    # Create an output widget to display the figures
    out = widgets.Output()

    # Display the slider
    display(slider, out)

    # Set the function to be called each time the slider's value changes
    slider.observe(on_slider_change, names='value')

    # # Call the function once to display the initial figure
    on_slider_change({'new': slider.value})
    
def call_ffmpeg_generate_video(fig_save_dir, prefix, suffix='.png', video_name="video.mp4", framerate=24, ):
    """
    Call ffmpeg to generate video from images. Assume {prefix}{index}{suffix} naming convention.
    """
    subprocess.run(["ls", "-l"]) 
    ffmpeg_command = [
        "ffmpeg", 
        "-framerate", str(framerate), 
        "-i", os.path.join(fig_save_dir, f"{prefix}%d{suffix}"), 
        "-c:v", "libx264", 
        "-pix_fmt", "yuv420p", 
        video_name
    ]
    subprocess.run(ffmpeg_command, check=True)