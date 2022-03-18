#include "gl_utils.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#define GL_LOG_FILE "gl.log"
#define MAX_SHADER_LENGTH 262144

int g_gl_width = 800;
int g_gl_height = 800;
GLFWwindow *g_window;
/*--------------------------------LOG FUNCTIONS-------------------------------*/
bool restart_gl_log() {
  FILE *file = fopen(GL_LOG_FILE, "w");
  if (!file) {
    fprintf(stderr,
            "ERROR: could not open GL_LOG_FILE log file %s for writing\n",
            GL_LOG_FILE);
    return false;
  }
  time_t now = time(NULL);
  char *date = ctime(&now);
  fprintf(file, "GL_LOG_FILE log. local time %s\n", date);
  fclose(file);
  return true;
}

bool gl_log(const char *message, ...) {
  va_list argptr;
  FILE *file = fopen(GL_LOG_FILE, "a");
  if (!file) {
    fprintf(stderr, "ERROR: could not open GL_LOG_FILE %s file for appending\n",
            GL_LOG_FILE);
    return false;
  }
  va_start(argptr, message);
  vfprintf(file, message, argptr);
  va_end(argptr);
  fclose(file);
  return true;
}

/* same as gl_log except also prints to stderr */
bool gl_log_err(const char *message, ...) {
  va_list argptr;
  FILE *file = fopen(GL_LOG_FILE, "a");
  if (!file) {
    fprintf(stderr, "ERROR: could not open GL_LOG_FILE %s file for appending\n",
            GL_LOG_FILE);
    return false;
  }
  va_start(argptr, message);
  vfprintf(file, message, argptr);
  va_end(argptr);
  va_start(argptr, message);
  vfprintf(stderr, message, argptr);
  va_end(argptr);
  fclose(file);
  return true;
}

void glfw_error_callback(int error, const char *description) {
  fputs(description, stderr);
  gl_log_err("%s\n", description);
}
// a call-back function
void glfw_framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  g_gl_width = width;
  g_gl_height = height;
  printf("width %i height %i\n", width, height);
  /* update any perspective matrices used here */
}

unsigned long frame_counter = 0;

void _update_fps_counter(GLFWwindow *window) {
  static double previous_seconds = glfwGetTime();
  static int frame_count;
  double current_seconds = glfwGetTime();
  double elapsed_seconds = current_seconds - previous_seconds;
  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = (double)frame_count / elapsed_seconds;
    char tmp[128];
    sprintf(tmp, "gl fps: %.2f (frames=%lu)", fps,
            (long unsigned int)frame_counter);
    glfwSetWindowTitle(window, tmp);
    frame_count = 0;
  }
  frame_count++;
}

/*-----------------------------------SHADERS----------------------------------*/
bool parse_file_into_str(const char *file_name, char *shader_str, int max_len) {
  shader_str[0] = '\0'; // reset string
  FILE *file = fopen(file_name, "r");
  if (!file) {
    gl_log_err("ERROR: opening file for reading: %s\n", file_name);
    return false;
  }
  int current_len = 0;
  char line[2048];
  strcpy(line, ""); // remember to clean up before using for first time!
  while (!feof(file)) {
    if (NULL != fgets(line, 2048, file)) {
      current_len += strlen(line); // +1 for \n at end
      if (current_len >= max_len) {
        gl_log_err(
            "ERROR: shader length is longer than string buffer length %i\n",
            max_len);
      }
      strcat(shader_str, line);
    }
  }
  if (EOF == fclose(file)) { // probably unnecesssary validation
    gl_log_err("ERROR: closing file from reading %s\n", file_name);
    return false;
  }
  return true;
}

void print_shader_info_log(GLuint shader_index) {
  int max_length = 2048;
  int actual_length = 0;
  char log[2048];
  glGetShaderInfoLog(shader_index, max_length, &actual_length, log);
  printf("shader info log for GL index %i:\n%s\n", shader_index, log);
  gl_log("shader info log for GL index %i:\n%s\n", shader_index, log);
}

bool create_shader(const char *file_name, GLuint *shader, GLenum type) {
  gl_log("creating shader from %s...\n", file_name);
  char shader_string[MAX_SHADER_LENGTH];
  parse_file_into_str(file_name, shader_string, MAX_SHADER_LENGTH);
  *shader = glCreateShader(type);
  const GLchar *p = (const GLchar *)shader_string;
  glShaderSource(*shader, 1, &p, NULL);
  glCompileShader(*shader);
  // check for compile errors
  int params = -1;
  glGetShaderiv(*shader, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    gl_log_err("ERROR: GL shader index %i did not compile\n", *shader);
    print_shader_info_log(*shader);
    return false; // or exit or something
  }
  gl_log("shader compiled. index %i\n", *shader);
  return true;
}

void print_programme_info_log(GLuint sp) {
  int max_length = 2048;
  int actual_length = 0;
  char log[2048];
  glGetProgramInfoLog(sp, max_length, &actual_length, log);
  printf("program info log for GL index %u:\n%s", sp, log);
  gl_log("program info log for GL index %u:\n%s", sp, log);
}

bool is_programme_valid(GLuint sp) {
  glValidateProgram(sp);
  GLint params = -1;
  glGetProgramiv(sp, GL_VALIDATE_STATUS, &params);
  if (GL_TRUE != params) {
    gl_log_err("program %i GL_VALIDATE_STATUS = GL_FALSE\n", sp);
    print_programme_info_log(sp);
    return false;
  }
  gl_log("program %i GL_VALIDATE_STATUS = GL_TRUE\n", sp);
  return true;
}

bool create_programme(GLuint vert, GLuint frag, GLuint *programme) {
  *programme = glCreateProgram();
  gl_log("created programme %u. attaching shaders %u and %u...\n", *programme,
         vert, frag);
  glAttachShader(*programme, vert);
  glAttachShader(*programme, frag);
  // link the shader programme. if binding input attributes do that before link
  glLinkProgram(*programme);
  GLint params = -1;
  glGetProgramiv(*programme, GL_LINK_STATUS, &params);
  if (GL_TRUE != params) {
    gl_log_err("ERROR: could not link shader programme GL index %u\n",
               *programme);
    print_programme_info_log(*programme);
    return false;
  }
  (is_programme_valid(*programme));
  // delete shaders here to free memory
  glDeleteShader(vert);
  glDeleteShader(frag);
  return true;
}

GLuint create_programme_from_files(const char *vert_file_name,
                                   const char *frag_file_name) {
  GLuint vert, frag, programme;
  (create_shader(vert_file_name, &vert, GL_VERTEX_SHADER));
  (create_shader(frag_file_name, &frag, GL_FRAGMENT_SHADER));
  (create_programme(vert, frag, &programme));
  return programme;
}
