from dataclasses import dataclass, field
import numpy as np

from ..base import RCModel


@dataclass
class Parametric_RC(RCModel):
    """Variable order RC model"""
    heavy_walls: list = field(default_factory=[0])  # (walls as _w)
    light_walls: list = field(default_factory=[0])  # (bridges as _b)
    irr: list = field(default_factory=[])  # inside is node -1, mass(if defined) is node bounds+1
    mass: int = 0  # (as _m)
    heat_contribution: list = field(default_factory=[])  # add coefficients for listed contribution

    # (indoor as _i)

    def __post_init__(self):

        _p = {k['name']: k for k in self.parameters}

        self.index = {}
        self.boundary = max(self.heavy_walls + self.light_walls) + 1
        self.heavy_walls = set(self.heavy_walls)
        self.light_walls = set(self.light_walls)
        self.irr = set(self.irr)

        self.states = [('TEMPERATURE', 'x_i', 'indoor space temperature')] \
                      + [('TEMPERATURE', 'x_w%d' % i, 'wall %d temperature' % i) for i in self.heavy_walls]
        self.states += [('TEMPERATURE', 'x_m', 'internal mass temperature')] if self.mass else []

        self.inputs = [('TEMPERATURE', 'Tb%d' % i, 'boundary %d temperature' % i) for i in range(self.boundary)] \
                      + [('POWER', 'Qh', 'HVAC system heat')]
        self.inputs += [('POWER', 'Qh%d' % i, 'Source %d heat' % i) for i in self.heat_contribution if not i == 0]

        if self.irr:
            self.inputs.append(('POWER', 'Qgh', 'global horizontal solar radiation'))

        self.outputs = [('TEMPERATURE', 'x_i', 'indoor space temperature')]

        self.params = [('MEASURE_DEVIATION', 'sigv', ''),
                       ('THERMAL_CAPACITY', 'C_i', 'indoor air, indoor walls, furnitures, etc. '),
                       ('STATE_DEVIATION', 'sigw_i', ''),
                       ('INITIAL_MEAN', 'x0_i', ''),
                       ('INITIAL_DEVIATION', 'sigx0_i', ''),
                       ]
        self.parameters = [_p['sigv'],
                           _p['C_i'],
                           _p['sigw_i'],
                           _p['x0_i'],
                           _p['sigx0_i'],
                           ]

        heavy_w_count = len(self.heavy_walls)
        for i in self.heavy_walls:
            self.params.extend([
                ('THERMAL_TRANSMITANCE', 'Ho_w%d' % i, 'between the boundary %d and the wall node' % i),
                ('THERMAL_TRANSMITANCE', 'Hi_w%d' % i, 'between the wall %d node and the indoor' % i),
                ('THERMAL_CAPACITY', 'C_w%d' % i, 'Wall %d' % i),
                ('STATE_DEVIATION', 'sigw_w%d' % i, ''),
                ('INITIAL_MEAN', 'x0_w%d' % i, ''),
                ('INITIAL_DEVIATION', 'sigx0_w%d' % i, ''),
            ])
            self.parameters.extend([
                _p.get('Ho_w%d' % i, {**_p['Ho_w'], 'name': 'Ho_w%d' % i}),
                _p.get('Hi_w%d' % i, {**_p['Hi_w'], 'name': 'Hi_w%d' % i}),
                _p.get('C_w%d' % i, {**_p['C_w'], 'name': 'C_w%d' % i}),
                _p.get('sigw_w%d' % i, {**_p['sigw_w'], 'name': 'sigw_w%d' % i}),
                _p.get('x0_w%d' % i, {**_p['x0_w'], 'name': 'x0_w%d' % i}),
                _p.get('sigx0_w%d' % i, {**_p['sigx0_w'], 'name': 'sigx0_w%d' % i}),
            ])
        (
            self.index['Ho_w'],
            self.index['Hi_w'],
            self.index['C_w'],
            self.index['sigw_w'],
            self.index['x0_w'],
            self.index['sigx0_w'],
        ) = [[j + i * 6 for i in range(heavy_w_count)] for j in range(6)]

        light_w_count = len(self.light_walls)
        for i in self.light_walls:
            self.params.append(
                ('THERMAL_TRANSMITANCE', 'Hb_w%d' % i, 'between the boundary %d and the indoor' % i))
            self.parameters.append(_p.get('Hb_w%d' % i, {**_p['Hb_w'], 'name': 'Hb_w%d' % i}))
        self.index['Hb_w'] = [6 * heavy_w_count + i for i in range(light_w_count)]

        if self.mass:
            self.params.extend([
                ('THERMAL_TRANSMITANCE', 'H_m', 'between the indoor and the internal mass'),
                ('THERMAL_CAPACITY', 'C_m', 'of the internal mass'),
                ('STATE_DEVIATION', 'sigw_m', ''),
                ('INITIAL_MEAN', 'x0_m', ''),
                ('INITIAL_DEVIATION', 'sigx0_m', ''),
            ])
            self.parameters.extend([
                _p.get('H_m%d' % i, {**_p['H_m'], 'name': 'H_m%d' % i}),
                _p.get('C_m%d' % i, {**_p['C_m'], 'name': 'C_m%d' % i}),
                _p.get('sigw_m%d' % i, {**_p['sigw_m'], 'name': 'sigw_m%d' % i}),
                _p.get('x0_m%d' % i, {**_p['x0_m'], 'name': 'x0_m%d' % i}),
                _p.get('sigx0_m%d' % i, {**_p['sigx0_m'], 'name': 'sigx0_m%d' % i}),
            ])
            (
                self.index['H_m'],
                self.index['C_m'],
                self.index['sigw_m'],
                self.index['x0_m'],
                self.index['sigx0_m'],
            ) = [6 * heavy_w_count + light_w_count + i for i in range(5)]

        for i in self.heat_contribution:
            self.params.append(('COEFFICIENT', 'Cv' + str(i + 1), 'source %d heating contribution' % i))
            self.parameters.append(_p.get('Cv%d' % i, {**_p['Cv'], 'name': 'Cv%d' % i}))
        self.index['Cv'] = [6 * heavy_w_count + light_w_count + 5 * self.mass + i for i in
                            range(len(self.heat_contribution))]

        for i in self.irr:
            self.params.append(('SOLAR_APERTURE', 'As' + str(i + 1), 'of the node %d (m2)' % i))
            self.parameters.append(_p.get('As%d' % i, {**_p['As'], 'name': 'As%d' % i}))
        self.index['As'] = [6 * heavy_w_count + light_w_count + 5 * self.mass + len(self.heat_contribution) + i for i in
                            range(len(self.irr))]

        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dR['sigv'][0, 0] = 1.0

        self.dQ['sigw_i'][1, 1] = 1.0
        self.dx0['x0_i'][1, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0
        for loc, i in enumerate(self.heavy_walls, start=1):
            self.dQ['sigw_w%d' % i][loc, loc] = 1.0
            self.dx0['x0_w%d' % i][loc, 0] = 1.0
            self.dP0['sigx0_w%d' % i][loc, loc] = 1.0
        if self.mass:
            loc += 1
            self.dQ['sigw_m'][loc, loc] = 1.0
            self.dx0['x0_m'][loc, 0] = 1.0
            self.dP0['sigx0_m'][loc, loc] = 1.0

    def update_continuous_ssm(self):
        (
            sigv,
            C_i,
            sigw_i,
            x0_i,
            sigx0_i,
            *unpack,
        ) = self.parameters.theta

        Ho_w = [unpack[i] for i in self.index['Ho_w']]
        Hi_w = [unpack[i] for i in self.index['Hi_w']]
        C_w = [unpack[i] for i in self.index['C_w']]
        sigw_w = [unpack[i] for i in self.index['sigw_w']]
        x0_w = [unpack[i] for i in self.index['x0_w']]
        sigx0_w = [unpack[i] for i in self.index['sigx0_w']]

        Hb_w = [unpack[i] for i in self.index['Hb_w']]
        Cv = [unpack[i] for i in self.index['Cv']]
        As = [unpack[i] for i in self.index['As']]

        (
            H_m,
            C_m,
            sigw_m,
            x0_m,
            sigx0_m,
        ) = [unpack[i] for i in [self.index['H_m'], self.index['C_m'], self.index['sigw_m'], self.index['x0_m'],
                                 self.index['sigx0_m']]] if self.mass else [None] * 5

        self._update_matrix(self.A, self.B, Hi_w, Ho_w, Hb_w, H_m, Cv, As)

        if self.mass:
            self.A = np.multiply(self.A, (1 / np.hstack((C_i, C_w, C_m)))[:, None])
            self.B = np.multiply(self.B, (1 / np.hstack((C_i, C_w, C_m)))[:, None])
            self.Q[self._diag] = np.hstack((sigw_i, sigw_w, sigw_m))
            self.x0[:, 0] = np.hstack((x0_i, x0_w, x0_m))
            self.P0[self._diag] = np.hstack((sigx0_i, sigx0_w, sigx0_m))
        else:
            self.A = np.multiply(self.A, (1 / np.hstack((C_i, C_w)))[:, None])
            self.B = np.multiply(self.B, (1 / np.hstack((C_i, C_w)))[:, None])
            self.Q[self._diag] = np.hstack((sigw_i, sigw_w))
            self.x0[:, 0] = np.hstack((x0_i, x0_w))
            self.P0[self._diag] = np.hstack((sigx0_i, sigx0_w))

        # matriciel ???
        # invC = np.diag(1 / np.append(C_i, C_w, C_m))
        # self.A = invC @ self.A
        # self.B = invC @ self.B

        self.R[0, 0] = sigv

    def _update_matrix(self, A, B, Hi_w, Ho_w, Hb_w, H_m, Cv, As):
        A[:] = 0
        for loc, i in enumerate(self.heavy_walls):
            A[0, 0] -= Hi_w[loc]
            A[0, i + 1] += Hi_w[loc]
            A[i + 1, 0] += Hi_w[loc]
            A[i + 1, i + 1] -= (Hi_w[loc] + Ho_w[loc])
            B[i + 1, i] = Ho_w[loc]

        for loc, i in enumerate(self.light_walls):
            A[0, 0] -= Hb_w[loc]
            B[0, i] = Hb_w[loc]

        if self.mass:
            A[0, 0] -= H_m
            A[0, -1] += H_m
            A[-1, -1] += H_m
            A[-1, -1] -= H_m

        B[0, self.boundary] = 1
        for loc, i in enumerate(self.heat_contribution):
            B[0, self.boundary + i] = Cv[loc]

        for loc, i in enumerate(self.irr):
            B[i + 1, -1] = As[loc]
        return A, B

    def update_continuous_dssm(self):
        (
            _,
            C_i,
            _,
            _,
            _,
            *unpack,
        ) = self.parameters.theta
        Ho_w = [unpack[i] for i in self.index['Ho_w']]
        Hi_w = [unpack[i] for i in self.index['Hi_w']]
        Hb_w = [unpack[i] for i in self.index['Hb_w']]
        C_w = [unpack[i] for i in self.index['C_w']]
        H_m, C_m = [unpack[i] for i in [self.index['H_m'], self.index['C_m']]] if self.mass else [None] * 2
        Cv = [unpack[i] for i in self.index['Cv']]
        As = [unpack[i] for i in self.index['As']]

        # A, B = self._update_matrix(self.A.copy(), self.B.copy(), Hi_w, Ho_w, Hb_w, H_m, Cv, As)
        A, B = self._update_matrix(
            np.zeros((self.nx, self.nx)),
            np.zeros((self.nx, self.nu)),
            Hi_w, Ho_w, Hb_w, H_m, Cv, As)
        self.dA['C_i'][0, :] = - A[0, :] / (C_i ** 2)
        self.dB['C_i'][0, :] = - B[0, :] / (C_i ** 2)

        for loc, i in enumerate(self.heavy_walls):
            self.dA['C_w%d' % i][i + 1, :] = - A[i + 1, :] / (C_w[loc] ** 2)
            self.dB['C_w%d' % i][i + 1, :] = - B[i + 1, :] / (C_w[loc] ** 2)

            self.dA['Hi_w%d' % i][0, 0] = -1 / C_i
            self.dA['Hi_w%d' % i][0, i + 1] = 1 / C_i
            self.dA['Hi_w%d' % i][i + 1, 0] = 1 / C_w[loc]
            self.dA['Hi_w%d' % i][i + 1, i + 1] = -1 / C_w[loc]

            self.dA['Ho_w%d' % i][i + 1, i + 1] = -1 / C_w[loc]
            self.dB['Ho_w%d' % i][i + 1, i] = 1 / C_w[loc]

        for i in self.light_walls:
            self.dA['Hb_w%d' % i][0, 0] = -1 / C_i
            self.dB['Hb_w%d' % i][0, i] = 1 / C_i

        if self.mass:
            self.dA['C_m'][-1, :] = - A[-1, :] / (C_m ** 2)
            self.dB['C_m'][-1, :] = - B[-1, :] / (C_m ** 2)

            self.dA['H_m'][0, 0] = -1 / C_i
            self.dA['H_m'][0, -1] = 1 / C_i
            self.dA['H_m'][-1, -1] = 1 / C_m
            self.dA['H_m'][-1, -1] = -1 / C_m

        for i in self.heat_contribution:
            self.dB['Cv%d' % i][0, self.boundary + 1] = 1 / C_i

        for i in self.irr:
            if i == -1:
                self.dB['As0'][0, -1] = 1 / C_i
            if self.mass and i == self.boundary:
                self.dB['As%d' % (self.boundary + 1)][-1, -1] = 1 / C_m
            else:
                self.dB['As%d' % (i + 1)][i + 1, -1] = 1 / C_w[i]
