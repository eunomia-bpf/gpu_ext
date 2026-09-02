/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef GPU_EXT_LOADER_IDENTITY_H
#define GPU_EXT_LOADER_IDENTITY_H

#include <errno.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

static inline int safe_loader_map_id(const struct bpf_map *map, __u32 *id)
{
	struct bpf_map_info info = {};
	__u32 size = sizeof(info);
	int fd;

	if (!map || !id)
		return -EINVAL;
	fd = bpf_map__fd(map);
	if (fd < 0)
		return fd;
	if (bpf_obj_get_info_by_fd(fd, &info, &size) != 0)
		return -errno;
	if (info.id == 0)
		return -ENOENT;
	*id = info.id;
	return 0;
}

static inline int safe_loader_link_id(const struct bpf_link *link, __u32 *id)
{
	struct bpf_link_info info = {};
	__u32 size = sizeof(info);
	int fd;

	if (!link || !id)
		return -EINVAL;
	fd = bpf_link__fd(link);
	if (fd < 0)
		return fd;
	if (bpf_obj_get_info_by_fd(fd, &info, &size) != 0)
		return -errno;
	if (info.id == 0)
		return -ENOENT;
	*id = info.id;
	return 0;
}

#endif /* GPU_EXT_LOADER_IDENTITY_H */
