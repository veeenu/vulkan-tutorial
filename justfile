run:
  cargo run --release

watch:
  cargo watch -c -x clippy

shaders:
  glslc src/shaders/shader.vert -o src/shaders/shader.vert.spv
  glslc src/shaders/shader.frag -o src/shaders/shader.frag.spv
