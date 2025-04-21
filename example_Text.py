import mujoco as mj
import mujoco.viewer
import numpy as np
import mujoco
import glfw
import OpenGL.GL as gl

# Load MuJoCo model
MODEL_XML = r"C:\Users\ethan\Documents\Edinburgh_Uni\HumanED\Shadow_gym2_project\Shadow_Gym2\shadow_gym\resources\hand\manipulate_block.xml"  # Replace with your own MuJoCo XML model
#Load the model and data structures
model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)

# --- Initialize GLFW ---
if not glfw.init():
    raise Exception("GLFW initialization failed")

window_width, window_height = 1200, 900
window = glfw.create_window(window_width, window_height, "MuJoCo Simulation with Overlay", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

# Make the OpenGL context current.
glfw.make_context_current(window)

# --- Create a MuJoCo rendering context ---
context = mujoco.MjrContext(model, 0)

# --- Create scene and camera objects for rendering ---
# Allocate a scene; the second parameter is the maximum number of geoms.
scene = mujoco.MjvScene(model, 1000)
# Create a camera; we'll use default parameters (you can adjust these as needed).
camera = mujoco.MjvCamera()
# Optionally adjust camera parameters for a better view of your robot:
camera.lookat = [0, 0, 0]
camera.distance = 2.0   # Increase or decrease to zoom in/out
camera.azimuth = 90     # Horizontal angle (in degrees)
camera.elevation = -20  # Vertical angle (in degrees)

# --- Main simulation loop ---
while not glfw.window_should_close(window):
    # Step the simulation.
    mujoco.mj_step(model, data)

    # Get the current framebuffer size and create a viewport.
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)

    # Update the scene from the simulation data.
    # The third parameter is for perturbations (set to None if unused).
    mujoco.mjv_updateScene(model, data, None, camera,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)

    # Clear the OpenGL buffers.
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Render the simulation scene (robot, environment, etc.)
    mujoco.mjr_render(viewport, scene, context)

    # Now render the overlay text on top of the simulation.
    big_text = "Hello, MuJoCo!"
    small_text = "Overlay text rendered via mjr_overlay"
    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,    # Font scale (150%)
                       mujoco.mjtGridPos.mjGRID_TOPLEFT,          # Position (top-left)
                       viewport,                                  # Viewport dimensions
                       big_text,                                  # Main text string
                       small_text,                                # Secondary text string
                       context)                                   # Rendering context

    # Swap the front and back buffers and process events.
    glfw.swap_buffers(window)
    glfw.poll_events()

# --- Clean up ---
glfw.terminate()