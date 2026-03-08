from texture.resize_textures import resize_main
from texture.corrupt_textures import corrupt_main
#from inpainting.batch_inpaint import inpaint_main
#from rendering.renderer import render_main


def main():
    resize_main()
    corrupt_main()

    #inpaint_main()
    #render_main()


if __name__ == "__main__":
    main()