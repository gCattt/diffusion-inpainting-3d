from src.texture.resize_textures import resize_main
from src.texture.corrupt_textures import corrupt_main
from src.rendering.renderer import renderer_main
from src.texture.mask_projection import mask_projection_main
from src.inpainting.batch_inpaint import inpaint_main
from src.texture.backprojection import backprojection_main
from src.rendering.render_final import render_final_main
from src.evaluation.compare_renders import compare_renders_main


def main(stage="full"):
    if stage in ["preprocess", "full"]:
        resize_main()
        corrupt_main()

    if stage in ["render", "full"]:
        renderer_main()
        mask_projection_main()

    if stage in ["inpaint", "full"]:
        inpaint_main()

    if stage in ["backproject", "full"]:
        backprojection_main()

    if stage in ["final", "full"]:
        render_final_main()
        compare_renders_main()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="full",
        choices=["preprocess", "render", "inpaint", "backproject", "final", "full"]
    )
    args = parser.parse_args()

    main(args.stage) # python -m src.main --stage <...>