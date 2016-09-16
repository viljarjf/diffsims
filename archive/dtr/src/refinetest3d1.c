/*
 * refinetest3d1.c
 *
 * Unit test for refinement procedure in 3D
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <math.h>

#include "basis.h"
#include "reflections.h"
#include "image.h"
#include "reproject.h"
#include "refine.h"
#include "utils.h"
#include "mapping.h"
#include "control.h"
#include "displaywindow.h"
#include "mrc.h"

static int check_cell(Basis *cell, Basis *cell_real) {
	
	int fail;
	
	printf("     Refinement    Actual\n");
	printf("---------------------------\n");
	printf("ax  %+8f     %+8f\n", cell->a.x/1e9, cell_real->a.x/1e9);
	printf("ay  %+8f     %+8f\n", cell->a.y/1e9, cell_real->a.y/1e9);
	printf("az  %+8f     %+8f\n", cell->a.z/1e9, cell_real->a.z/1e9);
	printf("bx  %+8f     %+8f\n", cell->b.x/1e9, cell_real->b.x/1e9);
	printf("by  %+8f     %+8f\n", cell->b.y/1e9, cell_real->b.y/1e9);
	printf("bz  %+8f     %+8f\n", cell->b.z/1e9, cell_real->b.z/1e9);
	printf("cx  %+8f     %+8f\n", cell->c.x/1e9, cell_real->c.x/1e9);
	printf("cy  %+8f     %+8f\n", cell->c.y/1e9, cell_real->c.y/1e9);
	printf("cz  %+8f     %+8f\n", cell->c.z/1e9, cell_real->c.z/1e9);
	
	fail = 0;
	if ( fabs(cell->a.x - cell_real->a.x) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: ax not determined correctly (got %8f, should be %8f)\n",
										cell->a.x/1e9, cell_real->a.x/1e9);
		fail = 1;
	} else {
		printf("ax is OK.\n");
	}
	if ( fabs(cell->a.y - cell_real->a.y) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: ay not determined correctly (got %8f, should be %8f)\n",
										cell->a.y/1e9, cell_real->a.y/1e9);
		fail = 1;
	} else {
		printf("ay is OK.\n");
	}
	if ( fabs(cell->a.z - cell_real->a.z) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: az not determined correctly (got %8f, should be %8f)\n",
										cell->a.z/1e9, cell_real->a.z/1e9);
		fail = 1;
	} else {
		printf("az is OK.\n");
	}
	if ( fabs(cell->b.x - cell_real->b.x) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: bx not determined correctly (got %8f, should be %8f)\n",
										cell->b.x/1e9, cell_real->b.x/1e9);
		fail = 1;
	} else {
		printf("bx is OK.\n");
	}
	if ( fabs(cell->b.y - cell_real->b.y) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: by not determined correctly (got %8f, should be %8f)\n",
										cell->b.y/1e9, cell_real->b.y/1e9);
		fail = 1;
	} else {
		printf("by is OK.\n");
	}
	if ( fabs(cell->c.x - cell_real->c.x) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: cx not determined correctly (got %8f, should be %8f)\n",
										cell->c.x/1e9, cell_real->c.x/1e9);
		fail = 1;
	} else {
		printf("cx is OK.\n");
	}
	if ( fabs(cell->c.z - cell_real->c.z) > 0.01e9 ) {
		fprintf(stderr, "refinetest3d: cz not determined correctly (got %8f, should be %8f)\n",
										cell->c.z/1e9, cell_real->c.z/1e9);
		fail = 1;
	} else {
		printf("cz is OK.\n");
	}
	
	printf("bz and cy not checked in this test.\n");
	
	return fail;

}	

int main(int argc, char *argv[]) {

	ControlContext *ctx;
	ReflectionList *reflections_real;
	Basis *cell_real;
	int fail;
	
	ctx = control_ctx_new();
	ctx->omega = deg2rad(0.0);
	ctx->lambda = lambda(300.0e3);	/* 300 keV */
	ctx->fmode = FORMULATION_PIXELSIZE;
	ctx->x_centre = 256;
	ctx->y_centre = 256;
	ctx->pixel_size = 5e7;
	image_add(ctx->images, NULL, 512, 512, deg2rad(00.0), ctx);
	ctx->images->images[0].features = image_feature_list_new();
	image_add(ctx->images, NULL, 512, 512, deg2rad(90.0), ctx);
	ctx->images->images[1].features = image_feature_list_new();
	//ctx->omega = deg2rad(90.0);
	//image_add(ctx->images, NULL, 512, 512, deg2rad(90.0), ctx);
	//ctx->images->images[2].features = image_feature_list_new();
	
	/* Fudge to avoid horrifying pointer-related death */
	ctx->dw = malloc(sizeof(DisplayWindow));
	ctx->dw->cur_image = 0;
	
	/* The "true" cell */
	cell_real = malloc(sizeof(Basis));
	cell_real->a.x = 5.0e9;  cell_real->a.y = 0.0e9;  cell_real->a.z = 0.0e9;
	cell_real->b.x = 0.0e9;  cell_real->b.y = 5.0e9;  cell_real->b.z = 0.0e9;
	cell_real->c.x = 0.0e9;  cell_real->c.y = 0.0e9;  cell_real->c.z = 5.0e9;
	/* The "real" reflections */
	reflections_real = reflection_list_from_cell(cell_real);
	ctx->images->images[0].features = reproject_get_reflections(&ctx->images->images[0], reflections_real);
	ctx->images->images[1].features = reproject_get_reflections(&ctx->images->images[1], reflections_real);
	//ctx->images->images[2].features = reproject_get_reflections(&ctx->images->images[2], reflections_real);
	
	/* The "model" cell to be refined */
	ctx->cell = malloc(sizeof(Basis));
	ctx->cell->a.x = 5.0e9;  ctx->cell->a.y = 0.1e9;  ctx->cell->a.z = 0.1e9;
	ctx->cell->b.x = 0.1e9;  ctx->cell->b.y = 5.0e9;  ctx->cell->b.z = 0.1e9;
	ctx->cell->c.x = 0.1e9;  ctx->cell->c.y = 0.1e9;  ctx->cell->c.z = 5.0e9;
	ctx->cell_lattice = reflection_list_from_cell(ctx->cell);
	ctx->images->images[0].rflist = reproject_get_reflections(&ctx->images->images[0], ctx->cell_lattice);
	reproject_partner_features(ctx->images->images[0].rflist, &ctx->images->images[0]);
	ctx->images->images[1].rflist = reproject_get_reflections(&ctx->images->images[1], ctx->cell_lattice);
	reproject_partner_features(ctx->images->images[1].rflist, &ctx->images->images[1]);
	//ctx->images->images[2].rflist = reproject_get_reflections(&ctx->images->images[2], ctx->cell_lattice);
	//reproject_partner_features(ctx->images->images[2].rflist, &ctx->images->images[2]);
	
	refine_do_cell(ctx);
	image_feature_list_free(ctx->images->images[0].rflist);
	image_feature_list_free(ctx->images->images[1].rflist);
	//image_feature_list_free(ctx->images->images[2].rflist);
	
	fail = check_cell(ctx->cell, cell_real);
	
	free(ctx);
	
	if ( fail ) return 1;
	
	printf("\n3D refinement test OK.\n");
	
	return 0;
	
}

/* Dummy function stubs */
#include "gtk-valuegraph.h"
void displaywindow_update_imagestack(DisplayWindow *dw) { };
void displaywindow_enable_cell_functions(DisplayWindow *dw, gboolean g) { };
void displaywindow_update(DisplayWindow *dw) { };
void displaywindow_error(const char *msg, DisplayWindow *dw) { };
guint gtk_value_graph_get_type() { return 0; };
GtkWidget *gtk_value_graph_new() { return NULL; };
void gtk_value_graph_set_data(GtkValueGraph *vg, double *data, unsigned int n) { };
