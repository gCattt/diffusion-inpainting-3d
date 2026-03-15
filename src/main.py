from src.texture.resize_textures import resize_main
from src.texture.corrupt_textures import corrupt_main
from src.rendering.renderer import renderer_main
from src.texture.mask_projection import mask_projection_main
from src.inpainting.batch_inpaint import inpaint_main


def main():
    resize_main()
    corrupt_main()

    renderer_main()
    mask_projection_main()

    inpaint_main()

    #render_main()


if __name__ == "__main__":
    main()