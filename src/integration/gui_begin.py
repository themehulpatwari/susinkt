import tkinter as tk
from tkinter import ttk, font
import math
import random
import sys
from typing import Optional

class BeautifulShortsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shorts Explorer ✨")
        self.root.geometry("1100x800")
        self.root.configure(bg='#0a0e27')
        
        # Make window slightly transparent for modern look (macOS)
        try:
            self.root.attributes('-alpha', 0.98)
        except:
            pass
        
        # Animation variables
        self.animation_id = None
        self.particles = []
        self.wave_offset = 0
        self.selected_category = None
        self.hover_scale = {}
        self.title_pulse = 0
        self.search_bar_glow = 0
        self.custom_input = None
        self.show_custom_input = False
        
        # Color scheme - Enhanced with more vibrant colors
        self.bg_color = '#0a0e27'
        self.accent_color = '#6c5ce7'
        self.accent_glow = '#a29bfe'
        self.secondary_color = '#74b9ff'
        self.text_color = '#ffffff'
        self.text_secondary = '#b2bec3'
        self.card_bg = '#1e2749'
        self.card_hover = '#2d3561'
        self.gradient_start = '#6c5ce7'
        self.gradient_end = '#fd79a8'
        
        # Categories with emojis and colors - Clean and minimal
        self.categories = [
            {"name": "Comedy", "emoji": "😂", "color": "#ff6b6b", "gradient": "#ee5a6f", "desc": "Laugh out loud"},
            {"name": "Music", "emoji": "🎵", "color": "#ff9ff3", "gradient": "#f368e0", "desc": "Musical beats"},
            {"name": "Gaming", "emoji": "🎮", "color": "#54a0ff", "gradient": "#5f27cd", "desc": "Gaming action"},
            {"name": "Cooking", "emoji": "🍳", "color": "#feca57", "gradient": "#ff9ff3", "desc": "Tasty recipes"},
        ]
        
        self.setup_ui()
        self.create_particles()
        self.animate()
        self.animate_title()
        
    def setup_ui(self):
        # Create canvas for background animations
        self.canvas = tk.Canvas(
            self.root, 
            width=1100, 
            height=800, 
            bg=self.bg_color,
            highlightthickness=0
        )
        self.canvas.place(x=0, y=0)
        
        # Create radial gradient circles for depth
        self.create_gradient_circles()
        
        # Main container with fade-in effect
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Title with pulsing effect
        self.title_label = tk.Label(
            self.main_frame,
            text="What do you want to watch right now?",
            font=('Helvetica Neue', 38, 'bold'),
            bg=self.bg_color,
            fg=self.text_color,
            justify='center'
        )
        self.title_label.pack(pady=(0, 40))
        
        # Custom input section with beautiful minimal design
        self.create_custom_input_section()
    
    def create_gradient_circles(self):
        """Create gradient circles for depth effect"""
        # Minimal - no circles for cleaner look
        pass
        
    def create_custom_input_section(self):
        """Create beautiful minimal custom input section"""
        input_container = tk.Frame(self.main_frame, bg=self.bg_color)
        input_container.pack(pady=(0, 0))
        
        # Aesthetic entry with rounded corners effect and subtle shadow
        self.entry_frame = tk.Frame(
            input_container,
            bg='#141824',
            highlightthickness=0,
            bd=0
        )
        self.entry_frame.pack(padx=5, pady=5)
        
        # Inner frame for better aesthetics
        inner_frame = tk.Frame(
            self.entry_frame,
            bg='#1e2433',
            highlightthickness=0,
            bd=0
        )
        inner_frame.pack(padx=2, pady=2)
        
        # Custom entry field - beautiful and clean
        self.custom_entry = tk.Entry(
            inner_frame,
            font=('Helvetica Neue', 20),
            bg='#1e2433',
            fg=self.text_color,
            insertbackground=self.accent_glow,
            bd=0,
            width=45,
            relief='flat',
            highlightthickness=0
        )
        self.custom_entry.pack(side='left', padx=30, pady=22)
        self.custom_entry.insert(0, "type what you want to watch...")
        self.custom_entry.configure(fg=self.text_secondary)
        
        # Bind events for placeholder behavior
        self.custom_entry.bind('<FocusIn>', self.on_entry_focus_in)
        self.custom_entry.bind('<FocusOut>', self.on_entry_focus_out)
        self.custom_entry.bind('<Return>', self.on_custom_submit)
        
        # Beautiful submit button with glow
        self.submit_btn = tk.Label(
            inner_frame,
            text="→",
            font=('Helvetica Neue', 32, 'bold'),
            bg='#1e2433',
            fg=self.accent_glow,
            cursor='hand2',
            padx=20
        )
        self.submit_btn.pack(side='left', padx=(0, 25), pady=22)
        self.submit_btn.bind('<Button-1>', self.on_custom_submit)
        self.submit_btn.bind('<Enter>', lambda e: self.submit_btn.configure(fg=self.gradient_end, font=('Helvetica Neue', 35, 'bold')))
        self.submit_btn.bind('<Leave>', lambda e: self.submit_btn.configure(fg=self.accent_glow, font=('Helvetica Neue', 32, 'bold')))
        
    def on_entry_focus_in(self, event):
        """Handle entry focus in"""
        if self.custom_entry.get() == "type what you want to watch...":
            self.custom_entry.delete(0, tk.END)
            self.custom_entry.configure(fg=self.text_color)
        # Add subtle glow on focus
        self.custom_entry.configure(bg='#252b42')
        self.submit_btn.configure(bg='#252b42')
    
    def on_entry_focus_out(self, event):
        """Handle entry focus out"""
        if self.custom_entry.get() == "":
            self.custom_entry.insert(0, "type what you want to watch...")
            self.custom_entry.configure(fg=self.text_secondary)
        # Remove glow
        self.custom_entry.configure(bg='#1e2433')
        self.submit_btn.configure(bg='#1e2433')
    
    def animate_entry_glow(self, active):
        """No glow animation needed - keeping it minimal"""
        pass
    
    def _cleanup(self):
        """Cleanup the window properly"""
        self.root.quit()
        self.root.destroy()
    
    def on_custom_submit(self, event):
        """Handle custom input submission"""
        text = self.custom_entry.get()
        if text and text != "type what you want to watch...":
            self.custom_input = text
            # Stop all animations
            if self.animation_id:
                self.root.after_cancel(self.animation_id)
            # Schedule destruction to happen after this event completes
            self.root.after(10, self._cleanup)
    
    def animate_title(self):
        """Animate the title with a subtle pulse effect"""
        self.title_pulse += 0.05
        # Create a pulsing effect by changing the font slightly
        pulse = abs(math.sin(self.title_pulse))
        color_val = int(200 + (pulse * 55))
        color = f'#{color_val:02x}{color_val:02x}{color_val:02x}'
        self.title_label.configure(fg=color)
        
        self.root.after(100, self.animate_title)
    
    def create_particles(self):
        """Create floating particles for background animation"""
        for _ in range(50):  # More particles for richer effect
            x = random.randint(0, 1100)
            y = random.randint(0, 800)
            size = random.randint(1, 3)
            speed = random.uniform(0.3, 1.5)
            color_choice = random.choice([self.accent_color, self.secondary_color, self.gradient_end])
            particle = {
                'x': x,
                'y': y,
                'size': size,
                'speed': speed,
                'color': color_choice,
                'id': None
            }
            self.particles.append(particle)
    
    def animate(self):
        """Enhanced main animation loop"""
        self.canvas.delete('particle')
        self.canvas.delete('wave')
        self.canvas.delete('glow')
        
        # Animate particles with color variety
        for particle in self.particles:
            particle['y'] -= particle['speed']
            if particle['y'] < -10:
                particle['y'] = 810
                particle['x'] = random.randint(0, 1100)
            
            # Draw particle with glow effect
            x, y = particle['x'], particle['y']
            size = particle['size']
            
            # Outer glow
            self.canvas.create_oval(
                x - size*2, y - size*2, x + size*2, y + size*2,
                fill='', outline=particle['color'],
                width=1, tags='glow'
            )
            
            # Inner particle
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=particle['color'],
                outline='',
                tags='particle'
            )
        
        # Draw multiple animated waves at the bottom
        self.wave_offset += 0.05
        
        # First wave
        points1 = []
        for x in range(0, 1100, 10):
            y = 720 + math.sin((x / 100) + self.wave_offset) * 15
            points1.extend([x, y])
        if len(points1) >= 4:
            points1.extend([1100, 800, 0, 800])
            self.canvas.create_polygon(
                points1,
                fill=self.accent_color,
                outline='',
                tags='wave',
                stipple='gray25'
            )
        
        # Second wave (offset)
        points2 = []
        for x in range(0, 1100, 10):
            y = 740 + math.sin((x / 80) + self.wave_offset + 1) * 20
            points2.extend([x, y])
        if len(points2) >= 4:
            points2.extend([1100, 800, 0, 800])
            self.canvas.create_polygon(
                points2,
                fill=self.gradient_end,
                outline='',
                tags='wave',
                stipple='gray12'
            )
        
        # Continue animation
        self.animation_id = self.root.after(50, self.animate)
    
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI error: {e}")
        finally:
            # Ensure cleanup happens even if something goes wrong
            try:
                self.root.quit()
            except:
                pass
            try:
                self.root.destroy()
            except:
                pass
        # Return the captured input (will be None if window was closed without input)
        return self.custom_input


def get_user_shorts_preference():
    """
    Display a beautiful GUI and get the user's preference for what shorts to watch.
    
    Returns:
        str: The user's input text, or None if the window was closed without input
    """
    root = tk.Tk()
    app = BeautifulShortsGUI(root)
    user_input = app.run()
    return user_input


if __name__ == "__main__":
    # When run as a standalone script, print the result to stdout
    result = get_user_shorts_preference()
    if result:
        print(result, end='')  # Print without newline for clean output
    sys.exit(0)
