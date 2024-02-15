run:
  cargo run --release

watch:
  cargo watch -c -x clippy

shaders:
  glslangValidator -V src/shaders/shader.vert -o src/shaders/shader.vert.spv
  glslangValidator -V src/shaders/shader.frag -o src/shaders/shader.frag.spv
