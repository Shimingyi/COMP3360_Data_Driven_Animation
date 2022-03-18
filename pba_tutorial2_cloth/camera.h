#ifndef CAMERA_H
#define CAMERA_H

#include "glad.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

// Defines several possible options for camera movement. Used as abstraction to
// stay away from window-system specific input methods
enum camera_movement { FORWARD, BACKWARD, LEFT, RIGHT };

// An abstract camera class that processes input and calculates the
// corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class camera_t {
public:
  // camera Attributes
  glm::vec3 m_pos;
  glm::vec3 m_fwd;
  glm::vec3 m_up;
  glm::vec3 m_right;
  glm::vec3 m_world_up;
  // euler Angles
  float m_yaw;
  float m_pitch;
  // camera options
  float m_speed;
  float m_sensitivity;
  float m_zoom;

  // constructor with vectors
  camera_t(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = -90.0f, // degrees
           float pitch = 0.0f)
      : // degrees
        m_fwd(glm::vec3(0.0f, 0.0f, -1.0f)), m_speed(2.5f), m_sensitivity(0.1f),
        m_zoom(45.0f) {
    m_pos = position;
    m_world_up = up;
    m_yaw = yaw;
    m_pitch = pitch;

    update_vectors();

    printf("camera pos: %f %f %f\n", m_pos.x, m_pos.y, m_pos.z);
    printf("camera fwd: %f %f %f\n", m_fwd.x, m_fwd.y, m_fwd.z);
    printf("camera right: %f %f %f\n", m_right.x, m_right.y, m_right.z);
  }

  // the view matrix calculated using Euler Angles and the LookAt Matrix
  glm::mat4 get_view_matrix() {
    return glm::lookAt(m_pos, m_pos + m_fwd, m_up);
  }

  // processes input received from any keyboard-like input system. Accepts
  // input parameter in the form of camera defined ENUM (to abstract it from
  // windowing systems)
  void process_keyboard_input(camera_movement movement, float dt) {
    float velocity = m_speed * dt;
    if (movement == FORWARD)
      m_pos += m_fwd * velocity;
    if (movement == BACKWARD)
      m_pos -= m_fwd * velocity;
    if (movement == LEFT)
      m_pos -= m_right * velocity;
    if (movement == RIGHT)
      m_pos += m_right * velocity;
  }

  // processes input received from a mouse input system. Expects the offset
  // value in both the x and y direction.
  void process_mouse_motion(float xoffset, float yoffset,
                            GLboolean constrainPitch = true) {
    xoffset *= m_sensitivity;
    yoffset *= m_sensitivity;

    m_yaw += xoffset;
    m_pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get
    // flipped
    if (constrainPitch) {
      if (m_pitch > 89.0f)
        m_pitch = 89.0f;
      if (m_pitch < -89.0f)
        m_pitch = -89.0f;
    }

    // update fwd, right and up Vectors using the updated Euler angles
    update_vectors();
  }

  // processes input received from a mouse scroll-wheel event. Only requires
  // input on the vertical wheel-axis
  void process_mouse_scroll(float yoffset) {
    m_zoom -= (float)yoffset;
    if (m_zoom < 1.0f)
      m_zoom = 1.0f;
    if (m_zoom > 45.0f)
      m_zoom = 45.0f;
  }

private:
  // calculates the front vector from the Camera's (updated) Euler Angles
  void update_vectors() {
    // calculate the new fwd vector
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_fwd = glm::normalize(front);
    // also re-calculate the right and up vector
    // normalize the vectors, because their length gets closer to 0 the more
    // you look up or down which results in slower movement.
    m_right = glm::normalize(glm::cross(m_fwd, m_world_up));
    m_up = glm::normalize(glm::cross(m_right, m_fwd));
  }
};
#endif