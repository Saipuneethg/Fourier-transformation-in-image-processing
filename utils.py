import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go

def load_image(uploaded_file):
    """Load an uploaded image and convert it to grayscale."""
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale for simplicity in FT
    return np.array(image)

def compute_ft(image):
    """Compute 2D Fourier Transform and magnitude spectrum."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum (adding 1 to avoid log(0))
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize purely for visualization purposes
    magnitude_spectrum_vis = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return fshift, magnitude_spectrum_vis

def inverse_ft(fshift):
    """Compute the inverse Fourier transform from the shifted frequency domain."""
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize back to 0-255 image range
    img_back_vis = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_back_vis

def apply_low_pass_filter(fshift, radius):
    """Apply a Low-Pass Filter by keeping the lower frequencies (center)."""
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask: 1 in the center, 0 outside
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), int(radius), 1, thickness=-1)
    
    # Apply mask
    fshift_filtered = fshift * mask
    return fshift_filtered, mask

def apply_high_pass_filter(fshift, radius):
    """Apply a High-Pass Filter by keeping the higher frequencies (edges) and blocking the center."""
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask: 0 in the center, 1 outside
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), int(radius), 0, thickness=-1)
    
    # Apply mask
    fshift_filtered = fshift * mask
    return fshift_filtered, mask

def plot_to_image(fig):
    """Utility to convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

def generate_sine_wave_plot():
    """Generate a plot showing the combination of low and high-frequency sine waves."""
    t = np.linspace(0, 1, 500)
    freq1, freq2 = 5, 20
    wave1 = np.sin(2 * np.pi * freq1 * t)
    wave2 = 0.5 * np.sin(2 * np.pi * freq2 * t)
    combined = wave1 + wave2

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    
    axs[0].plot(t, wave1, color='blue')
    axs[0].set_title(f'Low Frequency Sine Wave ({freq1} Hz)')
    axs[0].axis('off')
    
    axs[1].plot(t, wave2, color='green')
    axs[1].set_title(f'High Frequency Sine Wave ({freq2} Hz)')
    axs[1].axis('off')
    
    axs[2].plot(t, combined, color='red')
    axs[2].set_title('Combined Signal')
    axs[2].axis('off')
    
    plt.tight_layout()
    return plot_to_image(fig)

def plot_3d_magnitude_spectrum(fshift, title="3D Frequency Spectrum"):
    """
    Generate an interactive 3D surface plot of the frequency magnitude spectrum using Plotly.
    """
    # Calculate magnitude spectrum safely
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Downsample slightly if the image is too large to keep interactivity smooth
    max_dim = 256
    h, w = magnitude_spectrum.shape
    if h > max_dim or w > max_dim:
        scale = min(max_dim/h, max_dim/w)
        new_h, new_w = int(h * scale), int(w * scale)
        magnitude_spectrum = cv2.resize(magnitude_spectrum, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create coordinate grids
    y = np.arange(magnitude_spectrum.shape[0])
    x = np.arange(magnitude_spectrum.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Create the Plotly figure
    fig = go.Figure(data=[go.Surface(
        z=magnitude_spectrum, 
        x=X, 
        y=Y,
        colorscale='Viridis',
        showscale=False
    )])
    
    fig.update_layout(
        title=title,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='X Frequency',
            yaxis_title='Y Frequency',
            zaxis_title='Magnitude (dB)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2) # Default viewing angle
            )
        )
    )
    return fig

def create_sample_image():
    """Create a sample image for demonstrations."""
    # Create a 256x256 image
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (100, 200), 200, -1)
    cv2.circle(img, (180, 100), 40, 150, -1)
    
    # Add some high-frequency content (stripes/patterns)
    for i in range(0, 256, 10):
        cv2.line(img, (0, i), (256, i), 100, 1)
        
    # Add some noise
    noise = np.random.normal(0, 15, (256, 256)).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img
