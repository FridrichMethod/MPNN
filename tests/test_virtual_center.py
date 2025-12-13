"""Test virtual center geometry."""

import numpy as np
import torch


def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle between planes defined by (p0, p1, p2) and (p1, p2, p3).

    Returns angle in degrees.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)

    return torch.atan2(y, x) * 180.0 / np.pi


def compute_angle(a, b, c):
    """Compute angle between vectors b-a and b-c.

    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b

    cosine_angle = torch.sum(ba * bc, dim=-1) / (torch.norm(ba, dim=-1) * torch.norm(bc, dim=-1))
    angle = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))
    return angle * 180.0 / np.pi


def test_virtual_center_geometry():
    """Test virtual center geometry properties."""
    # Construct a simple backbone geometry
    # N at origin, Ca along x-axis, C in xy plane
    # Typical bond lengths: N-Ca ~ 1.46, Ca-C ~ 1.51, Ca-Cb ~ 1.53
    # Angles: N-Ca-C ~ 111 deg

    N = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    Ca = torch.tensor([[1.46, 0.0, 0.0]], dtype=torch.float32)

    # Place C such that N-Ca-C angle is reasonable
    # C is at (1.46 + 1.51 * cos(180-111), 1.51 * sin(180-111), 0)
    angle_rad = (180 - 111) * np.pi / 180
    Cx = 1.46 + 1.51 * np.cos(angle_rad)
    Cy = 1.51 * np.sin(angle_rad)
    C = torch.tensor([[Cx, Cy, 0.0]], dtype=torch.float32)

    # Add fake O atom (not used for V construction but needed for tensor unpacking)
    O = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    # Stack into input tensor shape [1, 4, 3] (Batch=1, Atoms=4, Coords=3)
    x = torch.stack([N, Ca, C, O], dim=1)

    # Extract components as in the model code
    N_vec, Ca_vec, C_vec, _ = (x[:, i, :] for i in range(4))

    # Replicate Cb construction logic from ProteinMPNN
    b = Ca_vec - N_vec
    c = C_vec - Ca_vec
    a = torch.cross(b, c, dim=-1)
    Cb_vec = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca_vec

    # Virtual center logic from implementation
    l_virtual = 3.06

    u_bc = Cb_vec - Ca_vec
    u_bc = u_bc / (torch.norm(u_bc, dim=-1, keepdim=True) + 1e-6)

    u_nc = N_vec - Ca_vec
    u_nc = u_nc / (torch.norm(u_nc, dim=-1, keepdim=True) + 1e-6)

    n_plane = torch.cross(u_nc, u_bc, dim=-1)
    n_plane = n_plane / (torch.norm(n_plane, dim=-1, keepdim=True) + 1e-6)

    # v_rot = - (n_plane x u_bc) based on theta=270 logic
    v_dir = -torch.cross(n_plane, u_bc, dim=-1)

    V_vec = Ca_vec + l_virtual * v_dir

    # --- Verify properties ---

    # 1. Length l = |V - Ca| should be 3.06
    dist_V_Ca = torch.norm(V_vec - Ca_vec, dim=-1)
    assert torch.allclose(dist_V_Ca, torch.tensor([l_virtual]), atol=1e-4)

    # 2. Angle theta = angle(V - Ca - Cb) should be 270 degrees
    # Note: standard angle function returns [0, 180].
    # 270 degrees usually implies directionality or reflex angle.
    # The construction rotates u_bc by 270 (or -90).
    # The geometric angle between vectors (V-Ca) and (Cb-Ca) should be 90 degrees.
    # Because cos(270) = 0.
    angle_V_Ca_Cb = compute_angle(V_vec, Ca_vec, Cb_vec)
    assert torch.allclose(angle_V_Ca_Cb, torch.tensor([90.0]), atol=1e-4)

    # 3. Dihedral tau = dihedral(V - Ca - Cb - N) should be 0 degrees
    # This implies V is cis to N with respect to Ca-Cb bond.
    dihedral_V_Ca_Cb_N = compute_dihedral(V_vec, Ca_vec, Cb_vec, N_vec)
    # Dihedral might be slightly off due to float precision but should be close to 0
    # Note: depending on definition order (V-Ca-Cb-N vs N-Ca-Cb-V), sign might flip, but 0 is 0.
    assert torch.abs(dihedral_V_Ca_Cb_N) < 1e-4
