cpdef long[::1] recursive_search(double[::1] ,double[::1] , double[:,:,::1] ,int ,
                    int ,const double[:,:,::1] ,int ,int ,double ,int[::1] ) 

cdef (long*,long,long,int) _recursive_search(int ,double[::1] ,double[::1] , double[:,:,::1] ,int , int ,
                          long* , long ,long ,const double[:,:,::1] ,long ,int,int ) nogil

cdef (long*,long,long,int) _recursive_search_limit(int ,double[::1] ,double[::1] , double[:,:,::1] ,int , int ,
                          long* , long ,long ,const double[:,:,::1] ,long ,int ,int,int ) nogil

cdef (long*,long,long,int) _recursive_search_limit_leaves(int ,double[::1] ,double[::1] , double[:,:,::1] ,int , int ,
                          long* , long ,long ,const double[:,:,::1] ,long ,int ,int,int ) nogil

cdef int check_intersect(double[:,:] ,double[:] ,double[:] ) nogil

cdef int check_contained(double[:,:] ,double[:] ,double[:] ) nogil

cdef long* resize_long_array(long* ,long, long ) nogil

cpdef long[::1] recursive_search_time(double[::1] ,double[::1] , double[:,:,::1] ,int ,
                    int ,const double[:,:,::1] ,int,double ,double[::1],int[::1])

cdef (int,int) _recursive_search_time(int ,double[::1] ,double[::1] , double[:,:,::1] ,int , int ,
                         int ,int* , int , int* , int ) nogil

cdef (long,double,double,long*,int,int) _filter_leaves(const double[:,:,::1] ,double[::1],double[::1],long* ,long ,int* ,int ,int* ,int ,
                                        long ,int, long ) nogil
