#include "spring.h"
#include "kdtree.h"
#include <string>
#include <set>
#include <algorithm>
#include <iostream>
#include "/usr/local/cuda-12.1/include/cuda_runtime_api.h"

using namespace std;

class Matrix
{
public:
	Matrix() {};
	Matrix(Mesh* _obj) :obj(_obj) {};
	~Matrix();
	void Insert_Matrix(unsigned int i, unsigned int j, unsigned int k,
		vector<pair<unsigned int, unsigned int>> &value_inline, map<pair<unsigned int, unsigned int>, float>& spring_length);

private:
	map<pair<unsigned int, unsigned int>, unsigned int>  mat;
	Mesh* obj;

};

Matrix::~Matrix()
{
}

void Matrix::Insert_Matrix(
	unsigned int i,
	unsigned int j,
	unsigned int k,
	vector<pair<unsigned int,unsigned int>> &value_inline,
	map<pair<unsigned int, unsigned int>, float>& spring_length)
{
	map<pair<unsigned int,unsigned int>,unsigned int>::iterator ite1 = mat.find(make_pair(i,j));
	map<pair<unsigned int,unsigned int>,unsigned int>::iterator ite2 = mat.find(make_pair(j,i));
	if(mat.end()!=ite1)
	{
		value_inline.push_back(make_pair(ite1->second,k));
		int v1,v2,v3,v4; //v3, v4 Ϊ���õ�
		v1 = ite1->second;
		v2 = k;
		v3 = ite1->first.first;
		v4 = ite1->first.second;
		glm::vec3 e1 = obj->vertices[v1] - obj->vertices[v3];
		glm::vec3 e2 = obj->vertices[v4] - obj->vertices[v3];
		glm::vec3 e3 = obj->vertices[v2] - obj->vertices[v3];

		float theta1 = glm::acos(glm::dot(e1, e2) / (glm::length(e1)*glm::length(e2) + FLT_EPSILON));
		float theta2 = glm::acos(glm::dot(e3, e2) / (glm::length(e3)*glm::length(e2) + +FLT_EPSILON));
		float length = glm::sqrt(glm::dot(e1, e1) + glm::dot(e3, e3) - 2 * glm::length(e1)*glm::length(e3)*glm::cos(theta1 + theta2));
		spring_length.insert(make_pair(make_pair(ite1->second, k), length));
		return;
	}
	if(mat.end()!=ite2)
	{
		value_inline.push_back(make_pair(ite2->second,k));
		int v1, v2, v3, v4; //v3, v4 Ϊ���õ�
		v1 = ite2->second;
		v2 = k;
		v3 = ite2->first.first;
		v4 = ite2->first.second;
		glm::vec3 e1 = obj->vertices[v1] - obj->vertices[v3];
		glm::vec3 e2 = obj->vertices[v4] - obj->vertices[v3];
		glm::vec3 e3 = obj->vertices[v2] - obj->vertices[v3];

		float theta1 = glm::acos(glm::dot(e1, e2) / (glm::length(e1)*glm::length(e2) + FLT_EPSILON));
		float theta2 = glm::acos(glm::dot(e3, e2) / (glm::length(e3)*glm::length(e2) + FLT_EPSILON));
		float length = glm::sqrt(glm::dot(e1, e1) + glm::dot(e3, e3) - 2 * glm::length(e1)*glm::length(e3)*glm::cos(theta1 + theta2));

		spring_length.insert(make_pair(make_pair(ite2->second, k), length));
		return;
	}

	mat.insert(make_pair(make_pair(i,j),k));

}


Springs::Springs(Mesh* spring_obj)
{
	if (spring_obj->get_obj_type() == SINGLE_LAYER_BOUNDARY)
	{
		get_cloth_boundary_spring(spring_obj); //������Ҫ����
		get_boundary_boundary_spring(spring_obj);
	}

	create_neigh(spring_obj);
}

Springs::~Springs()
{
	// not free gpu memory
}

void Springs::get_cloth_boundary_spring(Mesh* spring_obj)
{  
	//��һ�ν�����֮��Ӧ�ñ��������Ϣ���ı�����֡simulationʱʹ�õ�һ�ε�����
	//����˳��Ϊcloth1,cloth1_boundary,cloth2,cloth2_boundary...
	
	int g_start = 0;
	unsigned int *idx = new unsigned int[spring_obj->vertices.size()]; 
	for(int n=0;n<spring_obj->vertex_object.size();n+=2)  
	{
		unsigned int group_size = spring_obj->vertex_object[n].second;
		kdtree *kd = kd_create(3);
		
		for (int i=0;i<group_size;i++)   //Ϊ��Ƭ1����kdtree
		{
			idx[i+g_start] = i+g_start;
			int ret = kd_insert3f(kd, spring_obj->vertices[i+g_start].x,
				spring_obj->vertices[i+g_start].y,
				spring_obj->vertices[i+g_start].z,
				&idx[i+g_start]);
		}
		g_start += spring_obj->vertex_object[n].second;

		for (int i=0; i <spring_obj->vertex_object[n+1].second; i++)    //Ϊ�߽��еĵ������ڽ���
		{
			float kdpos[3];
			kdres *result = kd_nearest3f(kd, spring_obj->vertices[i+g_start].x,
				spring_obj->vertices[i+g_start].y,
				spring_obj->vertices[i+g_start].z);
			int *resultidx = (int*)kd_res_itemf(result, kdpos);
			cloth_boundary_springs.push_back(make_pair(i+g_start,*resultidx));
		}
		g_start += spring_obj->vertex_object[n+1].second;

		kd_free(kd);
	}
	delete[]idx;
	
}

void Springs::get_boundary_boundary_spring(Mesh* spring_obj)
{
	//���ѡȡN����Ϊ�߽�����Ƭ���ڵ�֮���������
	float max_dist = 0;
	const unsigned int NUM = 100;
	for (int i = 0; i<NUM; i++)
	{
		unsigned int idx1 = cloth_boundary_springs[i].first;
		unsigned int idx2 = cloth_boundary_springs[i].second;
		max_dist += glm::distance(spring_obj->vertices[idx1],spring_obj->vertices[idx2]);
	}
	max_dist /= NUM;
	cout << "�߽�����Ƭ���ڵ�֮��������룺" << max_dist << endl;

	////Ϊ�߽�֮�佨������:�̶�һ�飬��ʣ��boundary��������
	vector<pair<unsigned int,unsigned int>> start_end;
	int start = 0;
	for(int n=0;n<spring_obj->vertex_object.size();n += 2)
	{
		start += spring_obj->vertex_object[n].second;
		start_end.push_back(make_pair(start,start+spring_obj->vertex_object[n+1].second));
		start += spring_obj->vertex_object[n+1].second;
	}

	int *idx = new int[spring_obj->vertices.size()];
	for(int i=0;i<start_end.size();i++)
	{
		//��ǰ����Ϊ��iƬboundary
		//Ϊ��i�����������boundary����kdtree
		kdtree *kd = kd_create(3);
		for(int j=0;j<start_end.size();j++)
		{
			if(j == i) continue;
			for(int k=start_end[j].first;k<start_end[j].second;k++)
			{
				idx[k] = k;
				int ret = kd_insert3f(kd, spring_obj->vertices[k].x,
				spring_obj->vertices[k].y,
				spring_obj->vertices[k].z,
				&idx[k]);
			}
		}

		//��ʼ��������������
		for(int k=start_end[i].first;k<start_end[i].second;k++)
		{
			float kdpos[3];
			kdres *result = kd_nearest3f(kd, spring_obj->vertices[k].x,
				spring_obj->vertices[k].y,
				spring_obj->vertices[k].z);
			int *resultidx = (int*)kd_res_itemf(result, kdpos);

			if (glm::distance(spring_obj->vertices[k],spring_obj->vertices[*resultidx]) < max_dist*50
				&& glm::distance(spring_obj->vertices[k],spring_obj->vertices[*resultidx]) > 0) //��������жϣ���ֹ����
			{
				boundary_boundary_springs.push_back(make_pair(k,*resultidx));
			}
		}

		kd_free(kd);
	}
	delete[]idx;


	//map[boundary,cloth]
	map<unsigned int,unsigned int> map_spring;
	for(auto spring:cloth_boundary_springs)
		map_spring[spring.first] = spring.second;

	for(auto spring:boundary_boundary_springs)
	{
		boundary.insert(make_pair(map_spring[spring.first],map_spring[spring.second]));
	}








	////FR->_Piece2
	//vector<pair<unsigned int,unsigned int>> start_end;
	//int start = 0;
	//for(int n=0;n<spring_obj->vertex_object.size();n++)
	//{
	//	start_end.push_back(make_pair(start,start+spring_obj->vertex_object[n].second));
	//	start += spring_obj->vertex_object[n].second;
	//}
	//unsigned int *idx = new unsigned int[spring_obj->uni_vertices.size()]; 
	//kdtree *kd = k create(3);
	//for (int i=start_end[8].first;i<start_end[8].second;i++)   //Ϊ��Ƭ1����kdtree
	//	{
	//		idx[i] = i;
	//		int ret = k insert3f(kd, spring_obj->uni_vertices[i].x,
	//			spring_obj->uni_vertices[i].y,
	//			spring_obj->uni_vertices[i].z,
	//			&idx[i]);
	//	}

	//for(int i=start_end[1].first;i<start_end[1].second;i++)
	//{
	//	float kdpos[3];
	//		kdres *result = k nearest3f(kd, spring_obj->uni_vertices[i].x,
	//			spring_obj->uni_vertices[i].y,
	//			spring_obj->uni_vertices[i].z);
	//		int *resultidx = (int*)k res_itemf(result, kdpos);

	//		if (glm::distance(spring_obj->uni_vertices[i],spring_obj->uni_vertices[*resultidx]) < max_dist*20) //��������жϣ���ֹ����
	//		{
	//			boundary_boundary_springs.push_back(make_pair(i,*resultidx));
	//		}
	//}


}

void Springs::draw(Mesh* spring_obj)
{
	for (int i = 0; i < neigh1.size(); i++)
	{
		glm::vec4 v1 = spring_obj->vertices[i];
		for (int j = 0; j < neigh1[i].size(); j++)
		{
			glm::vec4 v2 = spring_obj->vertices[neigh1[i][j]];
	
			glBegin(GL_LINES);
			glColor3f(1.0, 1.0, 1.0);
			glVertex3f(v1.x, v1.y, v1.z);
			glVertex3f(v2.x, v2.y, v2.z);
			glEnd();
		}
	}

	for (int i = 0; i < neigh2.size(); i++)
	{
		glm::vec4 v1 = spring_obj->vertices[i];
		for (int j = 0; j < neigh2[i].size(); j++)
		{
			glm::vec4 v2 = spring_obj->vertices[neigh2[i][j]];
			glBegin(GL_LINES);
			glColor3f(1.0, 0, 0);
			glVertex3f(v1.x, v1.y, v1.z);
			glVertex3f(v2.x, v2.y, v2.z);
			glEnd();
		}
	}
}

void Springs::create_neigh(Mesh* spring_obj)
{

	//create neigh1 for each vertex
	neigh1.resize(spring_obj->vertices.size());
	for (int i = 0; i<spring_obj->faces.size(); i++)
	{
		unsigned int f[3];
		for (int j = 0; j<3; j++)
		{
			f[j] = spring_obj->faces[i].vertex_index[j];
		}
		if (!exist(neigh1[f[0]], f[1]))   //ȥ��neighbour���ظ����ڽӵ�
			neigh1[f[0]].push_back(f[1]);
		if (!exist(neigh1[f[0]], f[2]))
			neigh1[f[0]].push_back(f[2]);

		if (!exist(neigh1[f[1]], f[0]))
			neigh1[f[1]].push_back(f[0]);
		if (!exist(neigh1[f[1]], f[2]))
			neigh1[f[1]].push_back(f[2]);

		if (!exist(neigh1[f[2]], f[0]))
			neigh1[f[2]].push_back(f[0]);
		if (!exist(neigh1[f[2]], f[1]))
			neigh1[f[2]].push_back(f[1]);
	}

	for (int i = 0; i<cloth_boundary_springs.size(); i++)
	{
		unsigned int idx1 = cloth_boundary_springs[i].first;
		unsigned int idx2 = cloth_boundary_springs[i].second;

		neigh1[idx1].push_back(idx2);
		neigh1[idx2].push_back(idx1);
	}

	for (auto spring : boundary)
		neigh1[spring.first].push_back(spring.second);

	//create neigh2 for each vertex
	neigh2.resize(spring_obj->vertices.size());
	Matrix NR(spring_obj);   //Neighbour Relation
	vector<pair<unsigned int, unsigned int>> point_inline;  //�洢�������������ζԽǶ�������

	for (int i = 0; i<spring_obj->faces.size(); i++)
	{
		unsigned int f[3];
		for (int j = 0; j<3; j++)
		{
			f[j] = spring_obj->faces[i].vertex_index[j];
		}

		NR.Insert_Matrix(f[0], f[1], f[2], point_inline, bend_spring_length);
		NR.Insert_Matrix(f[0], f[2], f[1], point_inline, bend_spring_length);
		NR.Insert_Matrix(f[1], f[2], f[0], point_inline, bend_spring_length);
	}

	for (int i = 0; i<point_inline.size(); i++)
	{
		unsigned int fir = point_inline[i].first;
		unsigned int sec = point_inline[i].second;

		neigh2[fir].push_back(sec);
		neigh2[sec].push_back(fir);
	}

}

bool Springs::exist(const vector<unsigned int>& array, const unsigned int val)
{
	if (array.end() == find(array.begin(), array.end(), val))
		return false;
	else
		return true;
}

void Springs::CSR_structure_spring(Mesh* spring_obj, vector<unsigned int>& CSR_R, vector<s_spring>& CSR_C)
{
	CSR_spring(spring_obj, neigh1, CSR_R, CSR_C);
}

void Springs::CSR_bend_spring(Mesh* spring_obj, vector<unsigned int>& CSR_R, vector<s_spring>& CSR_C)
{
	CSR_spring(spring_obj, neigh2, CSR_R, CSR_C);
}


void Springs::CSR_spring(const Mesh* spring_obj, const vector<vector<unsigned int>>& neigh, vector<unsigned int>& CSR_R, vector<s_spring>& CSR_C)
{
	int start_idx = 0;
	for (int i = 0; i < neigh.size(); i++)
	{
		CSR_R.push_back(start_idx);
		start_idx += neigh[i].size();

		for (int j = 0; j < neigh[i].size(); j++)
		{
			s_spring tem_spring;
			tem_spring.end = neigh[i][j];
			tem_spring.original = glm::distance(spring_obj->vertices[i], spring_obj->vertices[tem_spring.end]);
			CSR_C.push_back(tem_spring);
		}
	}

	CSR_R.push_back(start_idx);   // ���ں�������ͳһ����
}