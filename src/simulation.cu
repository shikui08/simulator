
#include "simulator.h"
#include "spring.h"
#include "Mesh.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <fstream>

#include "watch.h"
#include "common.h"

using namespace std;

__global__ void compute_face_normal(glm::vec3* g_pos_in, unsigned int* cloth_index, const unsigned int cloth_index_size, glm::vec3* cloth_face);   //update cloth face normal
__global__ void verlet(glm::vec3 * g_pos_in, glm::vec3 * g_pos_old_in, glm::vec3 * g_pos_out, glm::vec3 * g_pos_old_out,
						unsigned int* CSR_R_STR, s_spring* CSR_C_STR, unsigned int* CSR_R_BD, s_spring* CSR_C_BD,
						D_BVH bvh, 
						//
						tmd::Vec3d*               	vertices,				
						std::array<int, 3>*  	triangles,				
						tmd::Node* 				 	nodes,					
						tmd::Vec3d* 				 	pseudonormals_triangles,	
						std::array<tmd::Vec3d, 3>*	pseudonormals_edges,		
						tmd::Vec3d* 				 	pseudonormals_vertices, 
						//
						glm::vec3* d_collision_force,
						const unsigned int NUM_VERTICES);  //verlet intergration
__global__ void update_vbo_pos(glm::vec4* pos_vbo, glm::vec3* pos_cur, const unsigned int NUM_VERTICES);
__global__ void compute_vbo_normal(glm::vec3* normals, unsigned int* CSR_R, unsigned int* CSR_C_adjface_to_vertex, glm::vec3* face_normal, const unsigned int NUM_VERTICES);

Simulator::Simulator()
{
	
}

Simulator::~Simulator()
{
	cudaFree(x_cur[0]);
	cudaFree(x_cur[1]);
	cudaFree(x_last[0]);
	cudaFree(x_last[1]);
	cudaFree(d_collision_force);
	cudaFree(d_CSR_R);
	cudaFree(d_CSR_C_adjface_to_vertex);
	cudaFree(d_face_normals);

	cudaFree(CSR_R_structure);
	cudaFree(CSR_R_bend);
	cudaFree(CSR_C_structure);
	cudaFree(CSR_C_bend);
	for(auto& c : cuda_bvh)
		delete c;
}

Simulator::Simulator(Mesh& sim_cloth, std::vector<Mesh>& body) :readID(0), writeID(1)
{
	init_cloth(sim_cloth);
	init_spring(sim_cloth);
	build_bvh(body);
}

void Simulator::init_cloth(Mesh& sim_cloth)
{
	// \d_vbo_array_resource points to cloth's array buffer  
	safe_cuda(cudaGraphicsGLRegisterBuffer(&d_vbo_array_resource, sim_cloth.vbo.array_buffer, cudaGraphicsMapFlagsWriteDiscard));   	//register vbo


	//set heap size, the default is 8M
	size_t heap_size = 256 * 1024 * 1024;  
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

	// Send the cloth's vertices to GPU
	const unsigned int vertices_bytes = sizeof(glm::vec3) * sim_cloth.vertices.size();
	safe_cuda(cudaMalloc((void**)&x_cur[0], vertices_bytes));			 // cloth vertices
	safe_cuda(cudaMalloc((void**)&x_cur[1], vertices_bytes));			 // cloth vertices
	safe_cuda(cudaMalloc((void**)&x_last[0], vertices_bytes));	 // cloth old vertices
	safe_cuda(cudaMalloc((void**)&x_last[1], vertices_bytes));	 // cloth old vertices
	safe_cuda(cudaMalloc((void**)&d_collision_force, sizeof(glm::vec3) * sim_cloth.vertices.size()));  //collision response force
	safe_cuda(cudaMemset(d_collision_force, 0, sizeof(glm::vec3) * sim_cloth.vertices.size()));    //initilize to 0

	x_cur_in = x_cur[readID];
	x_cur_out = x_cur[writeID];
	x_last_in = x_last[readID];
	x_last_out = x_last[writeID];

	vector<glm::vec3> tem_vertices(sim_cloth.vertices.size());
	for (int i=0;i< sim_cloth.vertices.size();i++)
	{
		tem_vertices[i] = glm::vec3(sim_cloth.vertices[i]);   // glm::vec4 -> glm::vec3
	}

	safe_cuda(cudaMemcpy(x_cur[0], &tem_vertices[0], vertices_bytes, cudaMemcpyHostToDevice));
	safe_cuda(cudaMemcpy(x_last[0], &tem_vertices[0], vertices_bytes, cudaMemcpyHostToDevice));

	//����normal��������ݣ�ÿ�����ڽӵ�������� + ÿ�����3���������
	vector<unsigned int> TEM_CSR_R;
	vector<unsigned int> TEM_CSR_C_adjface;
	get_vertex_adjface(sim_cloth, TEM_CSR_R, TEM_CSR_C_adjface);

	safe_cuda(cudaMalloc((void**)&d_CSR_R, sizeof(unsigned int) * TEM_CSR_R.size()));
	safe_cuda(cudaMalloc((void**)&d_CSR_C_adjface_to_vertex, sizeof(unsigned int) * TEM_CSR_C_adjface.size()));
	safe_cuda(cudaMemcpy(d_CSR_R, &TEM_CSR_R[0], sizeof(unsigned int) * TEM_CSR_R.size(), cudaMemcpyHostToDevice));
	safe_cuda(cudaMemcpy(d_CSR_C_adjface_to_vertex, &TEM_CSR_C_adjface[0], sizeof(unsigned int) * TEM_CSR_C_adjface.size(), cudaMemcpyHostToDevice));
	
	safe_cuda(cudaMalloc((void**)&d_face_normals, sizeof(glm::vec3) * sim_cloth.faces.size()));    //face normal

	safe_cuda(cudaGraphicsGLRegisterBuffer(&d_vbo_index_resource, sim_cloth.vbo.index_buffer, cudaGraphicsMapFlagsWriteDiscard));   	//register vbo
}

void Simulator::init_spring(Mesh& sim_cloth)
{
	cout << "build springs" << endl;
	// Construct structure and bend springs in GPU
	Springs springs(&sim_cloth);
	
	vector<unsigned int> TEM_CSR_R_structure, TEM_CSR_R_bend;
	vector<s_spring> TEM_CSR_C_structure, TEM_CSR_C_bend;

	springs.CSR_structure_spring(&sim_cloth, TEM_CSR_R_structure, TEM_CSR_C_structure);
	springs.CSR_bend_spring(&sim_cloth, TEM_CSR_R_bend, TEM_CSR_C_bend);

	safe_cuda(cudaMalloc((void**)&CSR_R_structure, TEM_CSR_R_structure.size() * sizeof(unsigned int)));
	safe_cuda(cudaMalloc((void**)&CSR_R_bend, TEM_CSR_R_bend.size() * sizeof(unsigned int)));
	safe_cuda(cudaMalloc((void**)&CSR_C_structure, TEM_CSR_C_structure.size() * sizeof(s_spring)));
	safe_cuda(cudaMalloc((void**)&CSR_C_bend, TEM_CSR_C_bend.size() * sizeof(s_spring)));

	safe_cuda(cudaMemcpy(CSR_R_structure, &TEM_CSR_R_structure[0], TEM_CSR_R_structure.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
	safe_cuda(cudaMemcpy(CSR_R_bend, &TEM_CSR_R_bend[0], TEM_CSR_R_bend.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
	safe_cuda(cudaMemcpy(CSR_C_structure, &TEM_CSR_C_structure[0], TEM_CSR_C_structure.size() * sizeof(s_spring), cudaMemcpyHostToDevice));
	safe_cuda(cudaMemcpy(CSR_C_bend, &TEM_CSR_C_bend[0], TEM_CSR_C_bend.size() * sizeof(s_spring), cudaMemcpyHostToDevice));
	
	cout << "springs build successfully!" << endl;
}


// __global__ void vvv(
// 	tmd::Vec3d*               	vertices,				
// 	std::array<int, 3>*  	triangles,				
// 	tmd::Node* 				 	nodes,					
// 	tmd::Vec3d* 				 	pseudonormals_triangles,	
// 	std::array<tmd::Vec3d, 3>*	pseudonormals_edges,		
// 	tmd::Vec3d* 				 	pseudonormals_vertices, 
// 	float* fooArray) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
  
//   auto ret = tmd::signed_distance(
// 	{i - 50,0,0},
// 	vertices,
// 	triangles,
// 	nodes,
// 	pseudonormals_triangles,
// 	pseudonormals_edges,
// 	pseudonormals_vertices
//   );
// 	fooArray[i] = ret.distance;
// }


void Simulator::build_bvh(std::vector<Mesh>& bodyvec)
{
	stop_watch watch;
	watch.start();
	for(auto & body : bodyvec) {
		Mesh bvh_body = body;   // for bvh consttruction
		bvh_body.vertex_extend(0.003);

		watch.start();
		cuda_bvh.emplace_back(new BVHAccel(bvh_body));

		{
			std::vector<tmd::Vec3d> vertices;
			std::vector<std::array<int, 3>> connectivity;
			for(auto& v : bvh_body.vertices)
				vertices.emplace_back(tmd::Vec3d(v[0], v[1], v[2]));
			for(auto& f : bvh_body.faces)
				connectivity.emplace_back(std::array<int, 3>({f.vertex_index[0], f.vertex_index[1], f.vertex_index[2]}));
			// std::vector<tmd::Vec3d> vertices = { { 1, -1, -1 }, { 1, 0, -1 }, { 1, 1, -1 }, { 1, -1, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 1, -1, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { -1, -1, -1 }, { -1, 0, -1 }, { -1, 1, -1 }, { -1, -1, 0 }, { -1, 0, 0 }, { -1, 1, 0 }, { -1, -1, 1 }, { -1, 0, 1 }, { -1, 1, 1 }, { 0, 1, -1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 0, -1, -1 }, { 0, -1, 0 }, { 0, -1, 1 }, { 0, 0, 1 }, { 0, 0, -1 } };
			// std::vector<std::array<int, 3>> connectivity = { { 0, 1, 3 }, { 1, 4, 3 }, { 1, 2, 4 }, { 2, 5, 4 }, { 3, 4, 6 }, { 4, 7, 6 }, { 4, 5, 7 }, { 5, 8, 7 }, { 12, 10, 9 }, { 12, 13, 10 }, { 13, 11, 10 }, { 13, 14, 11 }, { 15, 13, 12 }, { 15, 16, 13 }, { 16, 14, 13 }, { 16, 17, 14 }, { 14, 18, 11 }, { 14, 19, 18 }, { 19, 2, 18 }, { 19, 5, 2 }, { 17, 19, 14 }, { 17, 20, 19 }, { 20, 5, 19 }, { 20, 8, 5 }, { 9, 21, 12 }, { 21, 22, 12 }, { 21, 0, 22 }, { 0, 3, 22 }, { 12, 22, 15 }, { 22, 23, 15 }, { 22, 3, 23 }, { 3, 6, 23 }, { 15, 23, 16 }, { 23, 24, 16 }, { 23, 6, 24 }, { 6, 7, 24 }, { 16, 24, 17 }, { 24, 20, 17 }, { 24, 7, 20 }, { 7, 8, 20 }, { 10, 21, 9 }, { 10, 25, 21 }, { 25, 0, 21 }, { 25, 1, 0 }, { 11, 25, 10 }, { 11, 18, 25 }, { 18, 1, 25 }, { 18, 2, 1 } };
			// bvh_body.save("/tmp/1.obj");
			// HostTBuild bui;
			distInstBuildArr.emplace_back();
			distInstQueryArr.emplace_back();
			distInstBuildArr.back().construct(vertices, connectivity);
			distInstQueryArr.back().vertices                = distInstBuildArr.back().vertices; 
			distInstQueryArr.back().triangles               = distInstBuildArr.back().triangles; 
			distInstQueryArr.back().nodes                   = distInstBuildArr.back().nodes; 
			distInstQueryArr.back().pseudonormals_triangles = distInstBuildArr.back().pseudonormals_triangles; 
			distInstQueryArr.back().pseudonormals_edges     = distInstBuildArr.back().pseudonormals_edges; 
			distInstQueryArr.back().pseudonormals_vertices  = distInstBuildArr.back().pseudonormals_vertices; 
			distInstQueryArr.back().is_constructed          = true; 

			// std::cout << "BUILD\n";
			// HostTQuery mesh_distance_;

			// mesh_distance_.vertices                = bui.vertices; 
			// mesh_distance_.triangles               = bui.triangles; 
			// mesh_distance_.nodes                   = bui.nodes; 
			// mesh_distance_.pseudonormals_triangles = bui.pseudonormals_triangles; 
			// mesh_distance_.pseudonormals_edges     = bui.pseudonormals_edges; 
			// mesh_distance_.pseudonormals_vertices  = bui.pseudonormals_vertices; 
			// mesh_distance_.is_constructed          = true; 

			// for (int x = 0; x < 100; x += 1) {
			// 	const auto result = mesh_distance_.signed_distance({ x - 50, 0, 0 });
			// 	// const float exact = point_AABB_signed({ x - 50, 0, 0 }, { -1, -1, -1 }, { 1, 1, 1 });
			// 	std::cout << result.distance <<  std::endl;
			// }
			
			// for(int x = 0; x < 100; x += 1) {
			// 	auto ret = tmd::signed_distance(
			// 		{x - 50,0,0},
			// 		bui.vertices.data(),
			// 		bui.triangles.data(),
			// 		bui.nodes.data(),
			// 		bui.pseudonormals_triangles.data(),
			// 		bui.pseudonormals_edges.data(),
			// 		bui.pseudonormals_vertices.data()
			// 	);
			// 	std::cout << "? " << ret.distance << std::endl;
			// }
			// {
			// 	thrust::device_vector<tmd::Vec3d> vertices 								= bui.vertices; 
			// 	thrust::device_vector<std::array<int, 3>> triangles 					= bui.triangles; 
			// 	thrust::device_vector<tmd::Node> nodes 									= bui.nodes; 
			// 	thrust::device_vector<tmd::Vec3d> pseudonormals_triangles 				= bui.pseudonormals_triangles; 
			// 	thrust::device_vector<std::array<tmd::Vec3d, 3>> pseudonormals_edges 	= bui.pseudonormals_edges; 
			// 	thrust::device_vector<tmd::Vec3d> pseudonormals_vertices 				= bui.pseudonormals_vertices; 
			// 	thrust::device_vector<float> outdist;
			// 	outdist.resize(100);
			// 	printf(" %d %d  %d %d  %d %d; %lld %lld %lld %lld %lld %lld \n", 
			// 		vertices.size(),
			// 		triangles.size(),
			// 		nodes.size(),
			// 		pseudonormals_triangles.size(),
			// 		pseudonormals_edges.size(),
			// 		pseudonormals_vertices.size(),
			// 		thrust::raw_pointer_cast(vertices.data()),				
			// 		thrust::raw_pointer_cast(triangles.data()),				
			// 		thrust::raw_pointer_cast(nodes.data()),					
			// 		thrust::raw_pointer_cast(pseudonormals_triangles.data()),	
			// 		thrust::raw_pointer_cast(pseudonormals_edges.data()),		
			// 		thrust::raw_pointer_cast(pseudonormals_vertices.data())
			// 	);
			// 	auto error = cudaDeviceSetLimit(cudaLimitStackSize, 40 * 1024 * 1024);
			// 	vvv<<<dim3(10, 1, 1), dim3(10, 1, 1)>>>(
			// 		thrust::raw_pointer_cast(vertices.data()),				
			// 		thrust::raw_pointer_cast(triangles.data()),				
			// 		thrust::raw_pointer_cast(nodes.data()),					
			// 		thrust::raw_pointer_cast(pseudonormals_triangles.data()),	
			// 		thrust::raw_pointer_cast(pseudonormals_edges.data()),		
			// 		thrust::raw_pointer_cast(pseudonormals_vertices.data()), 
			// 		thrust::raw_pointer_cast(outdist.data()));
			// 	safe_cuda(cudaDeviceSynchronize());
			// 	thrust::copy_n(outdist.begin(), outdist.size(), std::ostream_iterator<float>(std::cout, ","));
			// }
		}

		watch.stop();
		cout << "bvh build done free time elapsed: " << watch.elapsed() << "us" << endl;
	}
}


void Simulator::simulate(Mesh* sim_cloth)
{
	//cuda kernel compute .........
	static int i = 0;
	static int substeps = 10;
	cuda_verlet((i++ % (cuda_bvh.size() * substeps)) / substeps, sim_cloth->vertices.size());

	cuda_update_vbo(sim_cloth);     // update array buffer for opengl

	swap_buffer();
}

void Simulator::get_vertex_adjface(Mesh& sim_cloth, vector<unsigned int>& CSR_R, vector<unsigned int>& CSR_C_adjface)
{
	vector<vector<unsigned int>> adjaceny(sim_cloth.vertices.size());
	for(int i=0;i<sim_cloth.faces.size();i++)
	{
		unsigned int f[3];
		for(int j=0;j<3;j++)
		{
			f[j] = sim_cloth.faces[i].vertex_index[j];
			adjaceny[f[j]].push_back(i);
		}
	}

	// i-th vertex adjacent face start_index = CSR_R[i], end_index = CSR_R[i+1]
	// then you can acess CSR_C_adjface[start_index->end_index]
	unsigned int start_idx = 0;
	for(int i=0;i<adjaceny.size();i++)
	{
		CSR_R.push_back(start_idx);
		start_idx += adjaceny[i].size();

		for(int j=0;j<adjaceny[i].size();j++)
		{
			CSR_C_adjface.push_back(adjaceny[i][j]);
		}
	}

	CSR_R.push_back(start_idx);
}

void Simulator::cuda_verlet(int frameidx, const unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	
	computeGridSize(numParticles, 512, numBlocks, numThreads);
	verlet <<< numBlocks, numThreads >>>(x_cur_in,x_last_in, x_cur_out, x_last_out,
										CSR_R_structure, CSR_C_structure, CSR_R_bend, CSR_C_bend,
										*(cuda_bvh[frameidx]->d_bvh), 
										//
										thrust::raw_pointer_cast(distInstQueryArr[frameidx].vertices.data()),				
										thrust::raw_pointer_cast(distInstQueryArr[frameidx].triangles.data()),				
										thrust::raw_pointer_cast(distInstQueryArr[frameidx].nodes.data()),					
										thrust::raw_pointer_cast(distInstQueryArr[frameidx].pseudonormals_triangles.data()),	
										thrust::raw_pointer_cast(distInstQueryArr[frameidx].pseudonormals_edges.data()),		
										thrust::raw_pointer_cast(distInstQueryArr[frameidx].pseudonormals_vertices.data()), 
										//
										d_collision_force,
										numParticles);

	// stop the CPU until the kernel has been executed
	safe_cuda(cudaDeviceSynchronize());
}

void Simulator::cuda_update_vbo(Mesh* sim_cloth)
{
	unsigned int numParticles = sim_cloth->vertices.size();

	size_t num_bytes;
	glm::vec4* d_vbo_vertex;           //point to vertex address in the OPENGL buffer
	glm::vec3* d_vbo_normal;           //point to normal address in the OPENGL buffer
	unsigned int* d_adjvertex_to_face;    // the order like this: f0(v0,v1,v2) -> f1(v0,v1,v2) -> ... ->fn(v0,v1,v2)
	
	safe_cuda(cudaGraphicsMapResources(1, &d_vbo_array_resource));
	safe_cuda(cudaGraphicsMapResources(1, &d_vbo_index_resource));
	safe_cuda(cudaGraphicsResourceGetMappedPointer((void **)&d_vbo_vertex, &num_bytes, d_vbo_array_resource));
	safe_cuda(cudaGraphicsResourceGetMappedPointer((void **)&d_adjvertex_to_face, &num_bytes, d_vbo_index_resource));

	d_vbo_normal = (glm::vec3*)((float*)d_vbo_vertex + 4 * sim_cloth->vertices.size() + 2 * sim_cloth->tex.size());   // ��ȡnormalλ��ָ��	

	unsigned int numThreads, numBlocks;

	// update vertex position
	computeGridSize(numParticles, 512, numBlocks, numThreads);
	update_vbo_pos << < numBlocks, numThreads >> > (d_vbo_vertex, x_cur_out, numParticles);
	safe_cuda(cudaDeviceSynchronize());  	// stop the CPU until the kernel has been executed

	// we need to compute face normal before computing vbo normal
	computeGridSize(sim_cloth->faces.size(), 512, numBlocks, numThreads);
	compute_face_normal << <numBlocks, numThreads >> > (x_cur_in, d_adjvertex_to_face, sim_cloth->vertex_indices.size(), d_face_normals);
	safe_cuda(cudaDeviceSynchronize());

	// update vertex normal
	computeGridSize(numParticles, 1024, numBlocks, numThreads);
	compute_vbo_normal << < numBlocks, numThreads >> > (d_vbo_normal, d_CSR_R, d_CSR_C_adjface_to_vertex, d_face_normals ,numParticles);
	safe_cuda(cudaDeviceSynchronize());

	safe_cuda(cudaGraphicsUnmapResources(1, &d_vbo_index_resource));
	safe_cuda(cudaGraphicsUnmapResources(1, &d_vbo_array_resource));
}

void Simulator::save(string file_name)
{
}

void Simulator::computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}

void Simulator::swap_buffer()
{
	swap(readID, writeID);

	x_cur_in = x_cur[readID];
	x_cur_out = x_cur[writeID];
	x_last_in = x_last[readID];
	x_last_out = x_last[writeID];
}

void Simulator::update_vertex(glm::vec3 new_value, const unsigned int idx)
{
	safe_cuda(cudaMemcpy(&x_cur_in[idx], &new_value[0], sizeof(glm::vec3), cudaMemcpyHostToDevice));
	safe_cuda(cudaMemcpy(&x_last_in[idx], &new_value[0], sizeof(glm::vec3), cudaMemcpyHostToDevice));
}

