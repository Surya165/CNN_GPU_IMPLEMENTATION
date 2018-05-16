kernel void add(global int *a,global int *b, global int *c)
{
	int gid = get_global_id(0);
	int gid2 = get_global_id(1);
	int gid3 = get_global_id(2);
	if(gid == 0)
	{
		a[gid] = 0;
	}
	else
	{
		a[gid] ++;
	}
	if(gid2 == 0)
	{
		b[gid] = 0;
	}
	else
	{
		b[gid] ++;
	}
	if(gid3 == 0)
	{
		c[gid3] ++;
	}
}
