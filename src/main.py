from src.texture.resize_textures import resize_main
from src.texture.corrupt_textures import corrupt_main
from src.rendering.renderer import renderer_main
from src.texture.mask_projection import mask_projection_main
from src.inpainting.batch_inpaint import inpaint_main
from src.texture.backprojection import backprojection_main
from src.rendering.render_final import render_final_main
from src.evaluation.compare_renders import compare_renders_main

def main():
    resize_main()
    corrupt_main()

    renderer_main()
    mask_projection_main()

    # inpaint_main()

    # backprojection_main()
    # render_final_main()

    # compare_renders_main()


if __name__ == "__main__":
    main()